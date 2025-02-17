import functools
import os
import random
import subprocess
import sys
import time
from multiprocessing.pool import ThreadPool
import argparse
import signal

mammoth_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(mammoth_path)
sys.path.append(mammoth_path + '/utils')

from utils import smart_joint

global active_jobs
global completed_jobs
global failed_jobs
active_jobs = {}
completed_jobs = {}
failed_jobs = {}

""""
command 
python scripts/local_launcher.py --file data/jobs/list_seq_cifar10_supcon_asym_sweep.txt --at_a_time 2 
python scripts/local_launcher.py --file data/jobs/list_seq_cifar10_supcon_sym_sweep.txt --at_a_time 2
python scripts/local_launcher.py --fil data/jobs/list_seq_tinyimg_supcon_asym_sweep.txt --at_a_time 2 --devices 6 7
python scripts/local_launcher.py --file data/jobs/list_seq_tinyimg_supcon_sym_sweep.txt --at_a_time 2 --devices 2 3


python scripts/local_launcher.py --file data/jobs/list_seq_cifar_supcon_task-il.txt --at_a_time 2 --devices 4

python scripts/local_launcher.py --file data/jobs/list_seq_cifar_supcon_class-il.txt --at_a_time 2 --devices 5

"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="file containing jobs")
    parser.add_argument("--redundancy", type=int, default=1, help="number of times to run each job")
    parser.add_argument("--at_a_time", type=int, default=1, help="number of jobs to run at a time")
    parser.add_argument("--start_from", type=int, default=0, help="start from job number")
    parser.add_argument("--reverse", action="store_true", help="reverse job order")
    parser.add_argument("--devices", type=int, nargs='+', help="list of GPU indices to run the jobs on (should be less or equal to at_a_time)")
    args = parser.parse_args()

    assert args.at_a_time >= 1, "at_a_time must be at least 1"
    assert args.redundancy >= 1, "redundancy must be at least 1"
    assert args.start_from >= 0, "start_from must be at least 0"
    assert len(args.devices) <= args.at_a_time, "too many devices provided, extra devices will not be used"

    jobs_list = [l for l in open(args.file, "r").read().splitlines() if l.strip() != "" and not l.startswith("#")][args.start_from:] * args.redundancy
    if args.reverse:
        jobs_list = list(reversed(jobs_list))
    jobname = args.file.strip().split("/")[-1].split("\\")[-1].split(".")[0]
    return args, jobs_list, jobname


def print_progress(basepath):
    global active_jobs
    global completed_jobs
    global failed_jobs
    # clean terminal
    print("\033c", end="")

    for job_index, (jobname, pid) in active_jobs.items():
        filename = smart_joint(basepath, f'{job_index + 1}.err')
        if not os.path.exists(filename):
            return

        print(f"Job {job_index + 1} ({jobname}) is running with pid {pid}:")

        # show last line of error, wait for job to end
        with open(filename, "r") as err:
            try:
                last_line = err.readlines()[-1]
            except BaseException:
                last_line = ""
            print(last_line.strip())

    print("Completed jobs:" + str(len(completed_jobs)))
    print("[" + " ".join([str(job_index + 1) for job_index, _ in completed_jobs.items()]) + "]")

    print("Failed jobs:" + str(len(failed_jobs)))
    print("[" + " ".join([str(job_index + 1) for job_index, _ in failed_jobs.items()]) + "]")


def run_job(jobdata, basedir, jobname, log=False):
    job, index, device_id = jobdata
    global active_jobs
    global completed_jobs
    global failed_jobs
    with open(smart_joint(basedir, f'{index + 1}.out'), "w") as out, open(smart_joint(basedir, f'{index + 1}.err'), "w") as err:
        p = subprocess.Popen(f"CUDA_VISIBLE_DEVICES={device_id} python utils/main.py " + job, shell=True, stdout=out, stderr=err)
        active_jobs[index] = (jobname, p.pid)
        p.wait()

        # check if job failed
        if p.returncode != 0:
            failed_jobs[index] = (jobname, p.pid)
        else:
            completed_jobs[index] = (jobname, p.pid)
    del active_jobs[index]


def main():
    args, jobs_list, jobname = parse_args()

    devices = args.devices
    device_count = len(devices)

    print("Running {} jobs on {} devices".format(len(jobs_list), device_count))
    time.sleep(2)

    # register signal handler to kill all processes on ctrl+c
    # giulia: you don't need this, it is very dangerous to kill all processes on a machine. 
    # def signal_handler(sig, frame):
    #     print('Killing all processes')
    #     if os.name == 'nt':
    #         os.system("taskkill /F /T /PID {}".format(os.getpid()))
    #     else:
    #         os.system("kill -9 -1")
    #     sys.exit(0)
    # signal.signal(signal.SIGINT, signal_handler)

    # create logs directory if it doesn't exist
    if not os.path.exists("logs"):
        os.makedirs("logs")
    basedir = smart_joint("logs", jobname) + time.strftime("_%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
    if not os.path.exists(basedir):
        os.makedirs(basedir)

    print("Jobname: {}".format(jobname))
    print("Logging to {}".format(basedir))

    # create thread pool
    pool = ThreadPool(processes=args.at_a_time)
    run_fn = functools.partial(run_job, basedir=basedir, jobname=jobname)
    # we distribute the jobs across available devices in a cyclic way
    result = pool.map_async(run_fn, [(job, i, devices[i % device_count]) for i, job in enumerate(jobs_list)], chunksize=1)

    # wait for all jobs to finish and print progress
    while not result._number_left == 0:
        print_progress(basedir)
        time.sleep(2)


if __name__ == '__main__':
    main()
