import torch
import torch.nn.functional as F
import math

import os
import pandas as pd
from typing import Dict, Tuple, Optional

# ---------- Utilities ----------
def safe_covariance(X: torch.Tensor, unbiased: bool = True) -> torch.Tensor:
    """
    Compute covariance matrix of X (n x d).
    Returns d x d tensor. If n == 1, returns zeros.
    """
    n = X.shape[0]
    mean = X.mean(dim=0, keepdim=True)
    Xm = X - mean
    if unbiased:
        return Xm.t().mm(Xm) / (n - 1)
    else:
        return Xm.t().mm(Xm) / n

def compute_class_stats(features: torch.Tensor, labels: torch.Tensor, eps: float = 1e-6
                       ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor], Dict[int, int]]:
    """
    Return (means, covs, counts) as dicts keyed by class label.
    means[c] -> (d,), covs[c] -> (d,d), counts[c] -> int
    """
    classes = torch.unique(labels, sorted=True).tolist()
    means, covs, counts = {}, {}, {}
    for c in classes:
        mask = (labels == c)
        Xc = features[mask]
        n = Xc.shape[0]
        counts[c] = int(n)
        if n == 0:
            means[c] = None
            covs[c] = None
        else:
            means[c] = Xc.mean(dim=0)
            covs[c] = safe_covariance(Xc)
            # regularize small numerical issues on cov when needed later
    return means, covs, counts

def pooled_covariance(covs: Dict[int, torch.Tensor], counts: Dict[int, int], eps: float = 1e-6
                     ) -> torch.Tensor:
    """
    Compute pooled covariance across classes using unbiased class covariances:
    Sigma_pooled = (sum_c (n_c - 1) Sigma_c) / (sum_c (n_c - 1))
    If denominator == 0 (e.g., all classes have n<=1), fall back to identity*eps.
    """
    d = None
    numerator = None
    denom = 0
    for c, cov in covs.items():
        n = counts.get(c, 0)
        if cov is None or n <= 1:
            continue
        if d is None:
            d = cov.shape[0]
            numerator = torch.zeros_like(cov)
        numerator += (n - 1) * cov
        denom += (n - 1)
    if denom <= 0 or numerator is None:
        if d is None:
            raise RuntimeError("Cannot determine feature dimension for pooled covariance.")
        return torch.eye(d) * eps
    pooled = numerator / denom
    return pooled

def kl_gaussian(mu0: torch.Tensor, cov0: torch.Tensor, mu1: torch.Tensor, cov1: torch.Tensor,
                eps: float = 1e-6) -> float:
    """
    KL(N0 || N1) for multivariate Gaussians.
    Adds eps to diagonal of covariances for numerical stability.
    Returns scalar python float.
    """
    d = mu0.shape[0]
    cov0 = cov0 + torch.eye(d, device=cov0.device, dtype=cov0.dtype) * eps
    cov1 = cov1 + torch.eye(d, device=cov1.device, dtype=cov1.dtype) * eps
    # Compute inverses and determinants via slogdet for stability
    sign1, logdet1 = torch.slogdet(cov1)
    sign0, logdet0 = torch.slogdet(cov0)
    if sign1 <= 0 or sign0 <= 0:
        # fallback small-regularization
        cov0 = cov0 + torch.eye(d, device=cov0.device, dtype=cov0.dtype) * (eps * 10)
        cov1 = cov1 + torch.eye(d, device=cov1.device, dtype=cov1.dtype) * (eps * 10)
        sign1, logdet1 = torch.slogdet(cov1)
        sign0, logdet0 = torch.slogdet(cov0)

    cov1_inv = torch.linalg.inv(cov1)
    trace_term = torch.trace(cov1_inv @ cov0).item()
    delta = (mu1 - mu0).unsqueeze(0)  # 1xd
    quad = (delta @ cov1_inv @ delta.t()).item()
    kl = 0.5 * (trace_term + quad - d + (logdet1 - logdet0).item())
    return float(kl)

# ---------- LDA predictor ----------
def lda_predict(X: torch.Tensor, means: Dict[int, torch.Tensor], shared_cov: torch.Tensor,
                priors: Optional[Dict[int, float]] = None) -> torch.Tensor:
    """
    Predict class labels for X using LDA with shared covariance.
    Returns tensor of predicted labels (same dtype as class keys).
    Linear discriminant: score_c = w_c^T x + b_c, where w_c = Sigma^{-1} mu_c, b_c = -0.5 mu_c^T Sigma^{-1} mu_c + log prior
    """
    device = X.device
    classes = sorted(means.keys())
    d = X.shape[1]
    # invert shared covariance
    cov = shared_cov + torch.eye(d, device=device, dtype=shared_cov.dtype) * 1e-6
    cov_inv = torch.linalg.inv(cov)

    ws = []
    bs = []
    for c in classes:
        mu = means[c]
        w = cov_inv @ mu
        b = -0.5 * (mu.unsqueeze(0) @ cov_inv @ mu.unsqueeze(1)).item()
        if priors is not None:
            b += math.log(priors.get(c, 1e-9))
        ws.append(w)
        bs.append(b)
    W = torch.stack(ws, dim=1)  # d x C
    B = torch.tensor(bs, device=device)  # C
    # scores: (N x d) @ (d x C) + C -> N x C
    scores = X @ W + B
    idx = torch.argmax(scores, dim=1)
    pred_labels = torch.tensor([classes[i] for i in idx.tolist()], device=device)
    return pred_labels

def qda_predict(X, means, covs, priors):
    """
    X: (N, d) tensor of examples
    means: dict[class] = (d,)
    covs: dict[class] = (d,d) covariance matrices (class-specific!)
    priors: dict[class] = float
    """
    classes = list(means.keys())
    N, d = X.shape
    scores = []

    for c in classes:
        mu = means[c]                             # (d,)
        Sigma = covs[c]                           # (d,d)
        pi = priors[c]

        # Add small jitter for numerical stability
        Sigma = Sigma + 1e-6 * torch.eye(d, device=Sigma.device)

        # Inverse and log-determinant
        L = torch.linalg.cholesky(Sigma)
        logdet = 2.0 * torch.sum(torch.log(torch.diag(L)))
        Sigma_inv = torch.cholesky_inverse(L)

        diff = X - mu
        quad = torch.einsum("nd,dd,nd->n", diff, Sigma_inv, diff)

        # Quadratic discriminant score
        score_c = -0.5 * (logdet + quad) + torch.log(torch.tensor(pi, device=X.device))
        scores.append(score_c)

    # Shape (num_classes, N) → (N,)
    scores = torch.stack(scores, dim=1)
    preds = scores.argmax(dim=1)

    return preds

# ---------- Pseudo-label assignment (nearest mean / EM) ----------
def assign_pseudo_labels_by_means(test_feats: torch.Tensor, class_means: Dict[int, torch.Tensor],
                                  normalize: bool = False, em_iters: int = 0) -> torch.Tensor:
    """
    Assign pseudo-labels to each test feature by nearest class mean (L2).
    Optionally run EM-like iterations: after assignment, recompute means using assigned test features,
    then reassign, repeat em_iters times.
    Returns tensor of labels for test_feats (same dtype as keys).
    """
    device = test_feats.device
    classes = sorted(class_means.keys())
    # initialize means matrix (C x d) for classes that exist; absent classes will be ignored by having NaNs
    valid_classes = [c for c in classes if class_means[c] is not None]
    if len(valid_classes) == 0:
        raise RuntimeError("No valid class means provided for pseudo-labeling.")

    means_mat = torch.stack([class_means[c] for c in valid_classes], dim=0)  # C_valid x d

    X = test_feats
    if normalize:
        X = F.normalize(X, dim=1)
        means_mat = F.normalize(means_mat, dim=1)

    for it in range(em_iters + 1):
        # compute L2 distances
        # dist^2 = x^2 - 2 x m^T + m^2  (we can broadcast)
        x2 = (X * X).sum(dim=1, keepdim=True)  # N x 1
        m2 = (means_mat * means_mat).sum(dim=1).unsqueeze(0)  # 1 x C
        xm = X @ means_mat.t()  # N x C
        d2 = x2 - 2 * xm + m2  # N x C
        assigned_idx = torch.argmin(d2, dim=1)  # N
        assigned_labels = [valid_classes[i] for i in assigned_idx.tolist()]
        assigned_labels = torch.tensor(assigned_labels, device=device)
        if it == em_iters:
            return assigned_labels
        # recompute means from assignments
        new_means = []
        for i, c in enumerate(valid_classes):
            mask = (assigned_idx == i)
            if mask.sum() == 0:
                # keep old mean
                new_means.append(means_mat[i])
            else:
                new_means.append(X[mask].mean(dim=0))
        means_mat = torch.stack(new_means, dim=0)
    # should never reach
    return assigned_labels

# ---------- Main pipeline ----------
def analyze_and_train_lda(buffer_features: torch.Tensor,
                          buffer_labels: torch.Tensor,
                          test_features: torch.Tensor,
                          test_labels: torch.Tensor,
                          evaluate_features: torch.Tensor,
                          evaluate_labels: torch.Tensor,
                          normalize_inputs: bool = False,
                          pseudolabel_em_iters: int = 0):
    """
    Runs the requested analysis and trains/evaluates the four LDA variants. Prints per-class stats and accuracies.
    """
    device = buffer_features.device if buffer_features is not None else test_features.device

    if normalize_inputs:
        buffer_features = F.normalize(buffer_features, dim=1) if buffer_features is not None else None
        test_features = F.normalize(test_features, dim=1)

    # Compute class stats
    buf_means, buf_covs, buf_counts = compute_class_stats(buffer_features, buffer_labels)
    test_means, test_covs, test_counts = compute_class_stats(test_features, test_labels)

    all_classes = sorted(set(list(buf_counts.keys()) + list(test_counts.keys())))
    d = test_features.shape[1]

    # Per-class metrics: KL(buffer || test), mean gap norm, cov difference Frobenius norm
    per_class_metrics = {}
    for c in all_classes:
        n_buf = buf_counts.get(c, 0)
        n_test = test_counts.get(c, 0)
        mu_buf = buf_means.get(c, None)
        mu_test = test_means.get(c, None)
        cov_buf = buf_covs.get(c, None)
        cov_test = test_covs.get(c, None)

        # handle absent classes by using degenerate small cov / zero mean
        if mu_buf is None:
            mu_buf = torch.zeros(d, device=device)
        if mu_test is None:
            mu_test = torch.zeros(d, device=device)
        if cov_buf is None:
            cov_buf = torch.eye(d, device=device) * 1e-6
        if cov_test is None:
            cov_test = torch.eye(d, device=device) * 1e-6

        # KL
        kl_bt = kl_gaussian(mu_buf, cov_buf, mu_test, cov_test, eps=1e-6)
        kl_tb = kl_gaussian(mu_test, cov_test, mu_buf, cov_buf, eps=1e-6)
        kl_sym = 0.5 * (kl_bt + kl_tb)

        mean_gap_norm = torch.norm(mu_buf - mu_test).item()
        cov_diff_norm = torch.norm(cov_buf - cov_test).item()  # Frobenius norm
        # rank of the dif ference between covariances (number of significant singular values)
        # use torch.linalg.matrix_rank (or torch.matrix_rank on older PyTorch)
        try:
            cov_rank_buf = torch.linalg.matrix_rank(cov_buf).item()
            cov_rank_test = torch.linalg.matrix_rank(cov_test).item()
        except Exception as e:
            cov_rank_buf = -1
            cov_rank_test = -1


        per_class_metrics[c] = {
            "n_buf": n_buf,
            "n_test": n_test,
            "kl_buf_to_test": kl_bt,
            "kl_test_to_buf": kl_tb,
            "kl_sym": kl_sym,
            "mean_gap_norm": mean_gap_norm,
            "cov_diff_fro": cov_diff_norm,
            "cov_rank_buf": cov_rank_buf,
            "cov_rank_test": cov_rank_test,
        }

    # Print per-class summary
    # # print("Per-class statistics (sample):")
    # for c in all_classes:
    #     m = per_class_metrics[c]
    #     # print(f"Class {c}: n_buf={m['n_buf']}, n_test={m['n_test']}, "
    #     #       f"KL(buf||test)={m['kl_buf_to_test']:.4f}, KL_sym={m['kl_sym']:.4f}, "
    #     #       f"||mean_gap||={m['mean_gap_norm']:.4f}, ||cov_diff||_F={m['cov_diff_fro']:.4f}")

    # Build pooled covariances for buffer and for test
    pooled_buf_cov = pooled_covariance(buf_covs, buf_counts)
    pooled_test_cov = pooled_covariance(test_covs, test_counts)

    # Priors (class frequencies on test set)
    total_test = sum(test_counts.values()) if len(test_counts) > 0 else 0
    priors = {c: (test_counts.get(c, 0) / total_test) if total_test > 0 else 1.0 for c in all_classes}

    # Prepare means dicts: ensure class keys present for LDA functions
    buf_means_complete = {c: (buf_means[c] if buf_means.get(c, None) is not None else torch.zeros(d, device=device)) for c in all_classes}
    test_means_complete = {c: (test_means[c] if test_means.get(c, None) is not None else torch.zeros(d, device=device)) for c in all_classes}

    buf_covs_complete = {c: (buf_covs[c] if buf_covs.get(c, None) is not None else torch.eye(d, device=device)) for c in all_classes}
    test_covs_complete = {c: (test_covs[c] if test_covs.get(c, None) is not None else torch.eye(d, device=device)) for c in all_classes}
    identity_class_covs = {c: torch.eye(d, device=device) for c in all_classes}

    # --- Classifier 1: buffer means, pooled covariance from buffer ---
    print("starting lda", flush=True)
    pred1 = lda_predict(evaluate_features, buf_means_complete, pooled_buf_cov, priors=priors)
    #pred1 = qda_predict(evaluate_features, buf_means_complete, buf_covs_complete, priors)
    acc1 = (pred1 == evaluate_labels).float().mean().item()
    print(f"\nLDA-1 (buffer means, pooled buffer cov) accuracy: {acc1:.4f}")

    # --- Classifier 2: test means, identity covariance ---
    identity_cov = torch.eye(d, device=device)
    pred2 = lda_predict(evaluate_features, test_means_complete, identity_cov, priors=priors)
    #pred2 = qda_predict(evaluate_features, test_means_complete, identity_class_covs, priors)
    acc2 = (pred2 == evaluate_labels).float().mean().item()
    print(f"LDA-2 (test means, identity cov) accuracy: {acc2:.4f}")

    # --- Classifier 3: test means, pooled test covariance ---
    pred3 = lda_predict(evaluate_features, test_means_complete, pooled_test_cov, priors=priors)
    #pred3 = qda_predict(evaluate_features, test_means_complete, test_covs_complete, priors)
    acc3 = (pred3 == evaluate_labels).float().mean().item()
    print(f"LDA-3 (test means, pooled test cov) accuracy: {acc3:.4f}")

    pred4= lda_predict(evaluate_features, buf_means_complete, identity_cov, priors=priors)
    #pred4 = qda_predict(evaluate_features, buf_means_complete, identity_class_covs, priors)
    acc4 = (pred4 == evaluate_labels).float().mean().item()
    print(f"LDA-4 (buffer means, id cov) accuracy: {acc4:.4f}")

    '''
    # --- Classifier 4: assign pseudo-labels to test (by nearest buffer means), then compute means/cov from those pseudo-labels and run LDA ---
    # Use buffer means as initialization for assignment
    init_buf_means = {c: buf_means.get(c, None) for c in all_classes}
    # remove classes with None to avoid errors in pseudo-label assignment
    nonnull_init = {c: m for c, m in init_buf_means.items() if m is not None}
    if len(nonnull_init) == 0:
        print("No buffer class means available for pseudo-labeling; skipping LDA-4.")
        pred4 = None
        acc4 = None
    else:
        pseudo_labels = assign_pseudo_labels_by_means(test_features, init_buf_means, normalize=False, em_iters=pseudolabel_em_iters)
        # compute means and covs from pseudo-labeled test features
        pseudo_means, pseudo_covs, pseudo_counts = compute_class_stats(test_features, pseudo_labels)
        # ensure complete set of classes: use pseudo means for classes that were present; otherwise zero
        pseudo_means_complete = {c: (pseudo_means[c] if pseudo_means.get(c, None) is not None else torch.zeros(d, device=device)) for c in all_classes}
        # pooled covariance from pseudo-labeled groups
        pooled_pseudo_cov = pooled_covariance(pseudo_covs, pseudo_counts)
        pred4 = lda_predict(evaluate_features, pseudo_means_complete, pooled_pseudo_cov, priors=priors)
        acc4 = (pred4 == evaluate_labels).float().mean().item()
        print(f"LDA-4 (pseudo-labeled test means, pooled pseudo cov), EM iter={pseudolabel_em_iters} accuracy: {acc4:.4f}")
    '''
    # Return useful objects for further inspection
    return {
        "per_class_metrics": per_class_metrics,
        "pooled_buf_cov": pooled_buf_cov,
        "pooled_test_cov": pooled_test_cov,
        "lda_accuracies": {
            "lda1_bufmeans_pooledbufcov": acc1,
            "lda2_testmeans_idcov": acc2,
            "lda3_testmeans_pooledtestcov": acc3,
            "lda4_pseudo_means_pooledcov": acc4
        },
        "predictions": {
            "pred1": pred1,
            "pred2": pred2,
            "pred3": pred3,
            "pred4": pred4
        }
    }


# parameter grids
datasets = ["seq-cifar100", "seq-cub200", "seq-tinyimg"]
buffer_sizes = [0, 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 50000, 6000, 8000, 10000, 100000, 240, 360, 480, 600, 720, 960, 1200, 12000]
seeds = [1000, 2000, 3000]

# where your features are stored
base_dir = "/cluster/scratch/dammeier/features_ICLR"

results = []

for dataset in datasets:
    for buffer_size in buffer_sizes:
        for seed in seeds:
            path = os.path.join(base_dir, dataset, f"bs{buffer_size}_s{seed}.pth")
            if not os.path.exists(path):
                print(f"[skip] {path} not found")
                continue
            try:
                data = torch.load(path, map_location="cpu")

                buffer_features = data["buffer_features"].to("cuda")
                buffer_labels = data["buffer_labels"].long().to("cuda")
                train_features = data["train_features"].to("cuda")
                train_labels = data["train_labels"].long().to("cuda")
                test_features = data["test_features"].to("cuda")
                test_labels = data["test_labels"].long().to("cuda")

                # run analysis
                res = analyze_and_train_lda(
                    buffer_features=buffer_features,
                    buffer_labels=buffer_labels,
                    test_features=train_features,
                    test_labels=train_labels,
                    evaluate_features=test_features,
                    evaluate_labels=test_labels,
                    normalize_inputs=False,
                    pseudolabel_em_iters=1
                )

                # flatten results into one row
                row = {
                    "dataset": dataset,
                    "buffer_size": buffer_size,
                    "seed": seed,
                }

                # add LDA accuracies
                row.update(res["lda_accuracies"])

                # add summary stats (averages across classes)
                metrics = res["per_class_metrics"]
                if len(metrics) > 0:
                    metric_list = list(metrics.values())  # preserve insertion order
                    cut = int(len(metric_list) * 0.9)
                    first90 = metric_list[:cut]

                    row["mean_KL_sym"] = sum(m["kl_sym"] for m in first90) / cut
                    row["mean_mean_gap"] = sum(m["mean_gap_norm"] for m in first90) / cut
                    row["mean_cov_diff"] = sum(m["cov_diff_fro"] for m in first90) / cut
                    row["mean_cov_rank_buf"] = sum(m["cov_rank_buf"] for m in first90) / cut
                    row["mean_cov_rank_test"] = sum(m["cov_rank_test"] for m in first90) / cut
                else:
                    row["mean_KL_sym"] = None
                    row["mean_mean_gap"] = None
                    row["mean_cov_diff"] = None
                    row["mean_cov_rank_buf"] = None
                    row["mean_cov_rank_test"] = None

                results.append(row)
                print(f"[done] {dataset}, bf={buffer_size}, seed={seed}")

            except Exception as e:
                print(f"[error] {path}: {e}")

# convert to DataFrame
df = pd.DataFrame(results)
print(df.head())

# save to csv
df.to_csv("lda_results.csv", index=False)
