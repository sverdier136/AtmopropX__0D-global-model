## test-github
test of github + training with git

# Workflow (1)

## Initial Setup
In `master`:
```bash
git cm      # checkout master
git pull origin master
git checkout -b <branch-name>
```

## Development
Make changes, then stage and commit:
```bash
git a  # add *
git commit -m "my changes"
```

## Keeping Up-to-Date
Rebase your branch with updates:
```bash
git fetch origin
git rebase origin/master    # rebom
```

## Finalizing the Feature
Merge back into `master`:
```bash
git checkout master
git pull origin master
git mff <feature-branch-name> # merge --no-ff <feature-branch-name>  
# --no-ff create a separate branch (no fast forward)
git push
```

## Tagging
Mark important points, such as releases:
```bash
git tag <version>
```

#
#
#
#
#

# Workflow (0)


### Initialisation
Dans ```master``` :
```git checkout -b <branch-name (ex : dev)>```

### Développement :
```git add *```
```git commit ```

### Mise à jour repo distant :
##### Mettre master à jour : 
```git checkout master``` ```git pull ```

##### Merge ```master``` et ```dev``` : 
```git checkout dev ``` ```git rebase master```

##### Résoudre les conflits : 
```git rebase --continue```

##### Push les modifs de ```dev``` sur ```origin/master``` : 
```git push origin dev:master```

##### Revenir à l'état normal : 
```git checkout master && git pull ```


# Workflow 1 détaillé :

from : https://gist.github.com/chalasr/fd195d83a0a01e4291a8

```
# everything is happy and up-to-date in master
git checkout master
git pull origin master

# let's branch to make changes
git checkout -b my-new-feature

# go ahead, make changes now.
$EDITOR file

# commit your (incremental, atomic) changes
git add -p
git commit -m "my changes"

# keep abreast of other changes, to your feature branch or master.
# rebasing keeps our code working, merging easy, and history clean.
git fetch origin
git rebase origin/my-new-feature
git rebase origin/master

# optional: push your branch for discussion (pull-request)
#           you might do this many times as you develop.
git push origin my-new-feature

# optional: feel free to rebase within your feature branch at will.
#           ok to rebase after pushing if your team can handle it!
git rebase -i origin/master

# merge when done developing.
# --no-ff preserves feature history and easy full-feature reverts
# merge commits should not include changes; rebasing reconciles issues
# github takes care of this in a Pull-Request merge
git checkout master
git pull origin master
git merge --no-ff my-new-feature


# optional: tag important things, such as releases
git tag 1.0.0-RC1 
```