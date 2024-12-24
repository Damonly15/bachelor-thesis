import socket
import requests

def check_internet_via_socket():
    try:
        # Try to connect to Google's public DNS server
        socket.create_connection(("8.8.8.8", 53), timeout=5)
        print("Internet connection is available (via socket).")
        return True
    except OSError:
        print("No internet connection (via socket).")
        return False

def check_internet_via_requests():
    try:
        # Try to make a request to the W&B API
        response = requests.get("https://api.wandb.ai", timeout=5)
        if response.status_code == 200:
            print("Internet connection is available (via requests).")
            return True
        else:
            print(f"Request failed with status code: {response.status_code}")
            return False
    except requests.ConnectionError:
        print("No internet connection (via requests).")
        return False
    except requests.Timeout:
        print("Connection timed out (via requests).")
        return False

# Run the checks
if __name__ == "__main__":
    has_internet_socket = check_internet_via_socket()
    has_internet_requests = check_internet_via_requests()

    if not has_internet_socket or not has_internet_requests:
        print("Internet access is not fully available.")
    else:
        print("Internet access confirmed.")


import wandb
wandb.login()
"""from backbone.CNN import CNN, count_parameters

model1 = CNN(16, 10)
model2 = CNN(32, 10)
model3 = CNN(64, 10)
model4 = CNN(112, 10)
model5 = CNN(176, 10)
model6 = CNN(240, 10)
model7 = CNN(368, 10)
model8 = CNN(512, 10)

print(count_parameters(model1))
print(count_parameters(model2))
print(count_parameters(model3))
print(count_parameters(model4))
print(count_parameters(model5))
print(count_parameters(model6))
print(count_parameters(model7))
print(count_parameters(model8))"""