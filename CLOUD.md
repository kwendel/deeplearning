### Google Cloud
How to connect with the Google Cloud Compute Engine.

## Linux
Reference: https://cloud.google.com/sdk/docs/quickstart-debian-ubuntu  

- Install the Google Cloud sdk
```
# Add the Cloud SDK distribution URI as a package source
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list

# Import the Google Cloud Platform public key
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -

# Update the package list and install the Cloud SDK
sudo apt-get update && sudo apt-get install google-cloud-sdk
```

- Next, initialize the SDK with: ```gcloud init``` in the terminal. This opens up a web browser where you must authorize the SDK to have access to your google account

- Follow the install steps in the command prompt. As a project, select the ```dl-project``` as the default. You can also enter your default time zone but it doesn't really matter for the end result.

- You can now run ```gcloud``` commands! To connect to the VM, start the instance and click on the ```SSH``` dropdown button. Click the ```View G-cloud command``` and enter this command in a terminal. This will make a SSH connection to the cloud instances.


## Windows
Reference: https://cloud.google.com/sdk/docs/quickstart-windows

- Install the Google Cloud SDK that can be found [here](https://cloud.google.com/sdk/docs/quickstart-windows)
- When the installer is finished, select the `Start Google Cloud SDK Shell` option to automatically configure the SDK.
- You can now enter `gcloud` commands in your terminal.


## Useful commands
Copy files between your local computer and the instances
- https://cloud.google.com/sdk/gcloud/reference/compute/scp







