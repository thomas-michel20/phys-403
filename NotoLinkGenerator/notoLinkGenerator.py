# PARAMETERS TO CHANGE:
git_https_url = "https://gitlab.epfl.ch/palermo/phys-403"
path_to_notebook = "Task3/step3.ipynb"


# Creation of the link
from nbgitpuller_link import Link
linker = Link(
    jupyterhub_url = "https://noto.epfl.ch",
    branch = "main",
    interface = "lab",
    repository_url = git_https_url,
    launch_path = path_to_notebook,
    )

# Display of the link
print(f"Your nbgitpuller link is:\n{linker.link}")
