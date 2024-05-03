## Effectively managing Multiple GitHub Accounts

For cases where you have both a personal GitHub (or similar) account and a company account, you can follow these steps to manage them effectively:

### 1. Generate SSH Keys

Create two SSH keys for the new machine, one for your professional account and one for your personal account:

```
ssh-keygen -t rsa -C "fernando.rodriguez@nielseniq.com"
```

This will generate two key pairs: `id_rsa` and `id_rsa.pub` for each account.

### 2. Create a Config File for Multiple Hosts

Create a configuration file named config in your ~/.ssh directory to allow for multiple hosts:

```
# Personal account
Host github.com-ferjorosa
	HostName github.com
	User git
	IdentityFile ~/.ssh/id_rsa_ferjorosa

# Professional account
Host github.com-niq
	HostName github.com
	User git
	IdentityFile ~/.ssh/id_rsa_niq
```

This file defines two hosts, github.com-ferjorosa and github.com-niq, each associated with the corresponding SSH key file.

### 3. Clone Repositories Using Different Accounts

Now you can clone repositories from different hosts using the appropriate SSH key:

```
git clone git@github.com-niq:niq-enterprise/ainn-omnisales-latam-data-prep.git
git clone git@github.com-ferjorosa:ferjorosa/learn-ebit.git
```
