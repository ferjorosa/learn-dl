[GitHub explanation](https://docs.github.com/es/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent?platform=linux)

* Step 1: Generate the SSH key
```
ssh-keygen -t "file_name" -C "your_email@example.com"
```

* Step 2: Add key to the ssh-agent

```
eval "$(ssh-agent -s)"

ssh-add ssh/private.key
```