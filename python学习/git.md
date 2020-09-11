##### 工作流程

![](../picture/1/141.png)

- 工作区：改动（增删文件和内容）
- 暂存区：输入命令：`git add 改动的文件名`，此次改动就放到了 ‘暂存区’，当对工作区修改或新增的文件执行 "git add" 命令时，暂存区的目录树被更新，同时工作区修改或新增的文件内容被写入到对象库中的一个新的对象中，而该对象的ID被记录在暂存区的文件索引中。
- 本地仓库：输入命令：`git commit` 此次修改的描述，此次改动就放到了 ’本地仓库’，每个 commit，我叫它为一个 ‘版本’。当执行提交操作（git commit）时，暂存区的目录树写到版本库（对象库）中，master 分支会做相应的更新。即 master 指向的目录树就是提交时暂存区的目录树。
- 远程仓库(简称：远程)：输入命令：`git push 远程仓库`，此次改动就放到了 ‘远程仓库’
- commit-id：输出命令：`git log`，最上面那行 `commit xxxxxx`，后面的字符串就是 commit-id

###### 将工作区加到暂存区

Git会将整个数据库储存在.git/目录下，Git提供了一个能够帮助你探索它的方法`git cat-file [-t] [-p]`，`-t`可以查看object的类型，`-p`可以查看object储存的具体内容。

git中由三种object，第一种blob类型，它只储存的是一个文件的内容，不包括文件名等其他信息。然后将这些信息经过SHA1哈希算法得到对应的哈希值，作为这个object在Git仓库中的唯一身份证。

```bash
#创建文件现有内容的一个副本。
git hash-boject -w filename #将文件现有的内容压缩成二进制文件，并保存到Git中。

#Git会在一个名叫“索引”的区域记录所有发生了变化的文件。然后等到所有的变更都结束后，将索引中的这些文件一起写入正式的版本历史记录中。
git update-index #文件名、二进制对象名（哈希值）以及索引中文件的访问权限。

git add --all #相当于针对当前项目中所有发生了变化的文件执行上述两个步骤
```

###### 将暂存区添加到本地仓库

索引保存发生了变化的文件信息。等到修改完成，所有这些信息都会被写入版本的历史记录中，这相当于生成一个当前项目的快照。项目的历史记录由不同时间点的项目快照组成。Git可以将项目恢复成任何一个快照。

第二种tree，它将当前的目录结构打了一个快照。从它储存的内容来看可以发现它储存了一个目录结构（类似于文件夹），以及每一个文件（或者子文件夹）的权限、类型、对应的身份证（SHA1值）、以及文件名。

第三种commit，它储存的是一个提交的信息，包括对应目录结构的快照tree的哈希值，上一个提交的哈希值，提交的作者以及提交的具体时间，最后是该提交的信息。

```bash
git write-tree #根据当前目录结构生成一个Git对象。
#git commit-tree可以将目录树对象写入到版本的历史记录中。
git commit-tree <tree-id>
```

##### 常用指令

| 指令                                                 | 作用                                                         |
| ---------------------------------------------------- | ------------------------------------------------------------ |
| `git help -g`                                        | 展示帮助信息                                                 |
| `git fetch --all && git reset --hard origin/master`  | 抛弃本地所有的修改，回到远程仓库的状态。                     |
| `git diff --cached`                                  | 输出**暂存区**和本地最近的版本 (commit) 的不同。             |
| `git diff HEAD`                                      | 输出**工作区**、**暂存区** 和本地最近的版本 (commit) 的不同。 |
| `git checkout -`                                     | 快速切换到上一个分支                                         |
| `git branch -vv`                                     | 展示本地分支关联远程仓库的情况                               |
| `git branch -u origin/mybranch`                      | 关联之后，`git branch -vv` 就可以展示关联的远程分支名了，同时推送到远程仓库直接：`git push`，不需要指定远程仓库了。 |
| `git branch -r`                                      | 列出所有远程分支                                             |
| `git branch -a`                                      | 列出所有远程分支                                             |
| `git remote show origin`                             | 查看远程分支和本地分支的对应关系                             |
| `git remote prune origin`                            | 远程删除了分支本地也想删除                                   |
| `git checkout -b <branch-name>`                      | 创建并切换到本地分支                                         |
| `git checkout -b <branch-name> origin/<branch-name>` | 从远程分支中创建并切换到本地分支                             |
| `git branch -d <local-branchname>`                   | 删除本地分支                                                 |
| `git push origin --delete <remote-branchname>`       | 删除远程分支                                                 |
| `git branch -m <new-branch-name>`                    | 删除远程分支                                                 |
| `git tag`                                            | 查看标签                                                     |
| `git push origin <local-version-number>`             | 推送标签到远程仓库                                           |
| `git checkout <file-name>`                           | 放弃工作区的修改                                             |
| `git checkout .`                                     | 放弃所有修改                                                 |
| `git revert <commit-id>`                             | 以新增一个 commit 的方式还原某一个 commit 的修改             |
| `git remote add origin <remote-url>`                 | 增加远程仓库                                                 |
| `git log`                                            | 查看 commit 历史                                             |
| `git remote`                                         | 列出所有远程仓库                                             |
|                                                      |                                                              |
| `git stash`                                          | 存储当前的修改，但不用提交 commit                            |

 如果你使用 `git status -s` 命令或 `git status --short`命令，你将得到一种更为紧凑的格式输出。新添加的未跟踪文件前面有 `??` 标记，新添加到暂存区中的文件前面有 `A` 标记，修改过的文件前面有 `M` 标记。 你可能注意到了 `M` 有两个可以出现的位置，出现在右边的 `M` 表示该文件被修改了但是还没放入暂存区，出现在靠左边的 `M` 表示该文件被修改了并放入了暂存区。

###### 回到某个 commit 的状态，并删除后面的 commit

和 revert 的区别：reset 命令会抹去某个 commit id 之后的所有 commit

```sh
git reset <commit-id>  #默认就是-mixed参数。

git reset –mixed HEAD^  #回退至上个版本，它将重置HEAD到另外一个commit,并且重置暂存区以便和HEAD相匹配，但是也到此为止。工作区不会被更改。

git reset –soft HEAD~3  #回退至三个版本之前，只回退了commit的信息，暂存区和工作区与回退之前保持一致。如果还要提交，直接commit即可  

git reset –hard <commit-id>  #彻底回退到指定commit-id的状态，暂存区和工作区也会变为指定commit-id版本的内容
```

###### 把 A 分支的某一个 commit，放到 B 分支上

```sh
git checkout <branch-name> && git cherry-pick <commit-id>
```

###### 给 git 命令起别名

```sh
git config --global alias.<handle> <command>
比如：git status 改成 git st，这样可以简化命令
git config --global alias.st status

$ git config --global alias.co checkout
$ git config --global alias.br branch
$ git config --global alias.ci commit
$ git config --global alias.st status
$ git config --global alias.unstage 'reset HEAD --'
$ git config --global alias.last 'log -1 HEAD'
$ git config --global alias.visual '!gitk'
# 我提交(commit)里的用户名和邮箱不对
git commit --amend --author "New Authorname <authoremail@mydomain.com>"
```

###### 展示所有 stashes
```sh
git stash list
```

###### 从 stash 中拿出某个文件的修改
```sh
git checkout <stash@{n}> -- <file-path>
```

###### 从远程仓库根据 ID，拉下某一状态，到本地分支

```sh
git fetch origin pull/<id>/head:<branch-name>
```

###### 展示所有 alias 和 configs

```sh
git config --local --list (当前目录)
git config --global --list (全局)
```

###### 撤销操作

提交完了才发现漏掉了几个文件没有添加，或者提交信息写错了。 此时，可以运行带有 `--amend` 选项的提交命令尝试重新提交：`git commit --amend`这个命令会将暂存区中的文件提交。 如果自上次提交以来你还未做任何修改，那么快照会保持不变，而你所修改的只是提交信息。文本编辑器启动后，可以看到之前的提交信息。 编辑后保存会覆盖原来的提交信息。最终你只会有一个提交 - 第二次提交将代替第一次提交的结果

使用 `git reset HEAD <file>...` 来取消暂存。撤销对文件的修改`git checkout -- <file>`。

`HEAD`指向的版本就是当前版本，上一个版本就是`HEAD^`，上上一个版本就是`HEAD^^`，当然往上100个版本写100个`^`比较容易数不过来，所以写成`HEAD~100`。因此，Git允许我们在版本的历史之间穿梭，使用命令`git reset --hard commit_id`。穿梭前，用`git log`可以查看提交历史，以便确定要回退到哪个版本。要重返未来，用`git reflog`查看命令历史，以便确定要回到未来的哪个版本。



###### git fetch

`git fetch` 命令与一个远程的仓库交互，并且将远程仓库中有但是在当前仓库的没有的所有信息拉取下来然后存储在你本地数据库中。

`git pull` 命令基本上就是 `git fetch` 和 `git merge` 命令的组合体，Git 从你指定的远程仓库中抓取内容，然后马上尝试将其合并进你所在的分支中。`git push` 命令用来与另一个仓库通信，计算你本地数据库与远程仓库的差异，然后将差异推送到另一个仓库中。 它需要有另一个仓库的写权限，因此这通常是需要验证的。`git remote` 命令是一个是你远程仓库记录的管理工具。 它允许你将一个长的 URL 保存成一个简写的句柄，例如 `origin` ，这样你就可以不用每次都输入他们了。 你可以有多个这样的句柄，`git remote`可以用来添加，修改，及删除它们。`git archive` 命令用来创建项目一个指定快照的归档文件。

首先要明确下，所有的版本控制系统，只能跟踪文本文件的改动，比如txt文件，网页，所有程序的代码等，Git也不列外，版本控制系统可以告诉你每次的改动，但是图片，视频这些二进制文件，虽能也能由版本控制系统管理，但没法跟踪文件的变化，只能把二进制文件每次改动串起来，也就是知道图片从1kb变成2kb，但是到底改了啥，版本控制也不知道。

##### 编辑提交(editting commits)

###### 我刚才提交了什么?

如果你用 `git commit -a` 提交了一次变化(changes)，而你又不确定到底这次提交了哪些内容。 你就可以用下面的命令显示当前`HEAD`上的最近一次的提交(commit):

```sh
git show
git log -n1 -p
```

###### 我想从一个提交(commit)里移除一个文件

通过下面的方法，从一个提交(commit)里移除一个文件:

```sh
$ git checkout HEAD^ myfile
$ git add -A
$ git commit --amend
```

###### 我想删除我的的最后一次提交(commit)

如果你需要删除推了的提交(pushed commits)，你可以使用下面的方法。可是，这会不可逆的改变你的历史，也会搞乱那些已经从该仓库拉取(pulled)了的人的历史。简而言之，如果你不是很确定，千万不要这么做。

```sh
$ git reset HEAD^ --hard
$ git push -f [remote] [branch]
```

如果你还没有推到远程, 把Git重置(reset)到你最后一次提交前的状态就可以了(同时保存暂存的变化):

```
(my-branch*)$ git reset --soft HEAD@{1}

```

这只能在没有推送之前有用. 如果你已经推了, 唯一安全能做的是 `git revert SHAofBadCommit`， 那会创建一个新的提交(commit)用于撤消前一个提交的所有变化(changes)； 或者, 如果你推的这个分支是rebase-safe的 (例如： 其它开发者不会从这个分支拉), 只需要使用 `git push -f`； 

###### 删除任意提交(commit)

同样的警告：不到万不得已的时候不要这么做.

```sh
$ git rebase --onto SHA1_OF_BAD_COMMIT^ SHA1_OF_BAD_COMMIT
$ git push -f [remote] [branch]
```

一般来说, **要避免强推**. 最好是创建和推(push)一个新的提交(commit)，而不是强推一个修正后的提交。后者会使那些与该分支或该分支的子分支工作的开发者，在源历史中产生冲突。

###### 我意外的做了一次硬重置(hard reset)，我想找回我的内容

如果你意外的做了 `git reset --hard`, 你通常能找回你的提交(commit), 因为Git对每件事都会有日志，且都会保存几天。

```sh
(master)$ git reflog
```

将会看到一个你过去提交的列表, 和一个重置的提交。 选择你想要回到的提交的SHA，再重置一次:

```sh
(master)$ git reset --hard SHA1234
```

##### 暂存(Staging)

###### 我需要把暂存的内容添加到上一次的提交(commit)

```sh
(my-branch*)$ git commit --amend
```

###### 我想要暂存一个新文件的一部分，而不是这个文件的全部

一般来说, 如果你想暂存一个文件的一部分, 你可这样做:

```sh
$ git add --patch filename.x
```

`-p` 简写。这会打开交互模式， 你将能够用 `s` 选项来分隔提交(commit)； 然而, 如果这个文件是新的, 会没有这个选择， 添加一个新文件时, 这样做:

```sh
$ git add -N filename.x
```

然后, 你需要用 `e` 选项来手动选择需要添加的行，执行 `git diff --cached` 将会显示哪些行暂存了哪些行只是保存在本地了。

###### 我想把在一个文件里的变化(changes)加到两个提交(commit)里

`git add` 会把整个文件加入到一个提交. `git add -p` 允许交互式的选择你想要提交的部分.

###### 我想把暂存的内容变成未暂存，把未暂存的内容暂存起来

多数情况下，你应该将所有的内容变为未暂存，然后再选择你想要的内容进行commit。
但假定你就是想要这么做，这里你可以创建一个临时的commit来保存你已暂存的内容，然后暂存你的未暂存的内容并进行stash。然后reset最后一个commit将原本暂存的内容变为未暂存，最后stash pop回来。

```sh
$ git commit -m "WIP"
$ git add .
$ git stash
$ git reset HEAD^
$ git stash pop --index 0
```

注意1: 这里使用`pop`仅仅是因为想尽可能保持幂等。
注意2: 假如你不加上`--index`你会把暂存的文件标记为为存储.这个[链接](https://stackoverflow.com/questions/31595873/git-stash-with-staged-files-does-stash-convert-staged-files-to-unstaged?answertab=active#tab-top) 解释得比较清楚。（不过是英文的，其大意是说，这是一个较为底层的问题，stash时会做2个commit，其中一个会记录index状态，staged的文件等东西，另一个记录worktree和其他的一些东西，如果你不在apply时加index，git会把两个一起销毁，所以staged里就空了）。

##### 未暂存(Unstaged)的内容

###### 我想把未暂存的内容移动到一个新分支

```sh
$ git checkout -b my-branch
```

###### 我想把未暂存的内容移动到另一个已存在的分支

```sh
$ git stash
$ git checkout my-branch
$ git stash pop
```

###### 我想丢弃本地未提交的变化(uncommitted changes)

如果你只是想重置源(origin)和你本地(local)之间的一些提交(commit)，你可以：

```sh
# one commit
(my-branch)$ git reset --hard HEAD^
# two commits
(my-branch)$ git reset --hard HEAD^^
# four commits
(my-branch)$ git reset --hard HEAD~4
# or
(master)$ git checkout -f
```

重置某个特殊的文件, 你可以用文件名做为参数:

```sh
$ git reset filename
```

###### 我想丢弃某些未暂存的内容

如果你想丢弃工作拷贝中的一部分内容，而不是全部。

签出(checkout)不需要的内容，保留需要的。

```sh
$ git checkout -p
# Answer y to all of the snippets you want to drop
```

另外一个方法是使用 `stash`， Stash所有要保留下的内容, 重置工作拷贝, 重新应用保留的部分。

```sh
$ git stash -p
# Select all of the snippets you want to save
$ git reset --hard
$ git stash pop
```

或者, stash 你不需要的部分, 然后stash drop。

```sh
$ git stash -p
# Select all of the snippets you don't want to save
$ git stash drop
```

##### 分支(Branches)

###### 我从错误的分支拉取了内容，或把内容拉取到了错误的分支

这是另外一种使用 `git reflog` 情况，找到在这次错误拉(pull) 之前HEAD的指向。

```sh
(master)$ git reflog
ab7555f HEAD@{0}: pull origin wrong-branch: Fast-forward
c5bc55a HEAD@{1}: checkout: checkout message goes here
```

重置分支到你所需的提交(desired commit):

```sh
$ git reset --hard c5bc55a
```

完成。

###### 我想扔掉本地的提交(commit)，以便我的分支与远程的保持一致

先确认你没有推(push)你的内容到远程。

`git status` 会显示你领先(ahead)源(origin)多少个提交:

```sh
(my-branch)$ git status
# On branch my-branch
# Your branch is ahead of 'origin/my-branch' by 2 commits.
#   (use "git push" to publish your local commits)
#
```

一种方法是:

```sh
(master)$ git reset --hard origin/my-branch
```

###### 我需要提交到一个新分支，但错误的提交到了master

在master下创建一个新分支，不切换到新分支,仍在master下:

```sh
(master)$ git branch my-branch
```

把master分支重置到前一个提交:

```sh
(master)$ git reset --hard HEAD^
```

`HEAD^` 是 `HEAD^1` 的简写，你可以通过指定要设置的`HEAD`来进一步重置。

或者, 如果你不想使用 `HEAD^`, 找到你想重置到的提交(commit)的hash(`git log` 能够完成)， 然后重置到这个hash。 使用`git push` 同步内容到远程。

例如, master分支想重置到的提交的hash为`a13b85e`:

```sh
(master)$ git reset --hard a13b85e
HEAD is now at a13b85e
```

签出(checkout)刚才新建的分支继续工作:

```sh
(master)$ git checkout my-branch
```

###### 我想保留来自另外一个ref-ish的整个文件

假设你正在做一个原型方案(原文为working spike (see note)), 有成百的内容，每个都工作得很好。现在, 你提交到了一个分支，保存工作内容:

```sh
(solution)$ git add -A && git commit -m "Adding all changes from this spike into one big commit."
```

当你想要把它放到一个分支里 (可能是`feature`, 或者 `develop`), 你关心是保持整个文件的完整，你想要一个大的提交分隔成比较小。

假设你有:

  * 分支 `solution`, 拥有原型方案， 领先 `develop` 分支。
  * 分支 `develop`, 在这里你应用原型方案的一些内容。

我去可以通过把内容拿到你的分支里，来解决这个问题:

```sh
(develop)$ git checkout solution -- file1.txt
```

这会把这个文件内容从分支 `solution` 拿到分支 `develop` 里来:

```sh
# On branch develop
# Your branch is up-to-date with 'origin/develop'.
# Changes to be committed:
#  (use "git reset HEAD <file>..." to unstage)
#
#        modified:   file1.txt
```

然后, 正常提交。

###### 我想删除上游(upstream)分支被删除了的本地分支

一旦你在github 上面合并(merge)了一个pull request, 你就可以删除你fork里被合并的分支。 如果你不准备继续在这个分支里工作, 删除这个分支的本地拷贝会更干净，使你不会陷入工作分支和一堆陈旧分支的混乱之中。

```sh
$ git fetch -p
```

###### 我不小心删除了我的分支

有些时候还是可能删除了还没有推到远程的分支。 误删了某个分支如何还原分支中的文件。在这时候你应该想起了`reflog`, 它存储了仓库(repo)里面所有动作的历史。

```
git reflog #找到分支的<commit id>
git checkout -b my-branch-help # 新建一个分支并切换到此分支
git reset --hard 4e3cd85<commit id>
```

###### 我想删除一个分支

删除一个远程分支:

```sh
(master)$ git push origin --delete my-branch
```

你也可以:

```sh
(master)$ git push origin :my-branch
```

删除一个本地分支:

```sh
(master)$ git branch -D my-branch
```

###### 我想从别人正在工作的远程分支签出(checkout)一个分支

首先, 从远程拉取(fetch) 所有分支:

```sh
(master)$ git fetch --all
```

假设你想要从远程的`daves`分支签出到本地的`daves`

```sh
(master)$ git checkout --track origin/daves
Branch daves set up to track remote branch daves from origin.
Switched to a new branch 'daves'
```

(`--track` 是 `git checkout -b [branch] [remotename]/[branch]` 的简写)

这样就得到了一个`daves`分支的本地拷贝, 任何推过(pushed)的更新，远程都能看到.

##### Rebasing 和合并(Merging)

###### 我想撤销rebase/merge

你可以合并(merge)或rebase了一个错误的分支, 或者完成不了一个进行中的rebase/merge。 Git 在进行危险操作的时候会把原始的HEAD保存在一个叫ORIG_HEAD的变量里, 所以要把分支恢复到rebase/merge前的状态是很容易的。

```sh
(my-branch)$ git reset --hard ORIG_HEAD
```

###### 我已经rebase过, 但是我不想强推(force push)

不幸的是，如果你想把这些变化(changes)反应到远程分支上，你就必须得强推(force push)。 是因你快进(Fast forward)了提交，改变了Git历史, 远程分支不会接受变化(changes)，除非强推(force push)。这就是许多人使用 merge 工作流, 而不是 rebasing 工作流的主要原因之一， 开发者的强推(force push)会使大的团队陷入麻烦。使用时需要注意，一种安全使用 rebase 的方法是，不要把你的变化(changes)反映到远程分支上, 而是按下面的做:

```sh
(master)$ git checkout my-branch
(my-branch)$ git rebase -i master
(my-branch)$ git checkout master
(master)$ git merge --ff-only my-branch
```

###### 我需要组合(combine)几个提交(commit)

假设你的工作分支将会做对于 `master` 的pull-request。 一般情况下你不关心提交(commit)的时间戳，只想组合 *所有* 提交(commit) 到一个单独的里面, 然后重置(reset)重提交(recommit)。 确保主(master)分支是最新的和你的变化都已经提交了, 然后:

```sh
(my-branch)$ git reset --soft master
(my-branch)$ git commit -am "New awesome feature"
```

如果你想要更多的控制, 想要保留时间戳, 你需要做交互式rebase (interactive rebase):

```sh
(my-branch)$ git rebase -i master
```

如果没有相对的其它分支， 你将不得不相对自己的`HEAD` 进行 rebase。 例如：你想组合最近的两次提交(commit), 你将相对于`HEAD~2` 进行rebase， 组合最近3次提交(commit), 相对于`HEAD~3`, 等等。

```sh
(master)$ git rebase -i HEAD~2
```

在你执行了交互式 rebase的命令(interactive rebase command)后, 你将在你的编辑器里看到类似下面的内容:

```vim
pick a9c8a1d Some refactoring
pick 01b2fd8 New awesome feature
pick b729ad5 fixup
pick e3851e8 another fix

# Rebase 8074d12..b729ad5 onto 8074d12
#
# Commands:
#  p, pick = use commit
#  r, reword = use commit, but edit the commit message
#  e, edit = use commit, but stop for amending
#  s, squash = use commit, but meld into previous commit
#  f, fixup = like "squash", but discard this commit's log message
#  x, exec = run command (the rest of the line) using shell
#
# These lines can be re-ordered; they are executed from top to bottom.
#
# If you remove a line here THAT COMMIT WILL BE LOST.
#
# However, if you remove everything, the rebase will be aborted.
#
# Note that empty commits are commented out
```

所有以 `#` 开头的行都是注释, 不会影响 rebase.

然后，你可以用任何上面命令列表的命令替换 `pick`, 你也可以通过删除对应的行来删除一个提交(commit)。

例如, 如果你想 **单独保留最旧(first)的提交(commit),组合所有剩下的到第二个里面**, 你就应该编辑第二个提交(commit)后面的每个提交(commit) 前的单词为 `f`:

```vim
pick a9c8a1d Some refactoring
pick 01b2fd8 New awesome feature
f b729ad5 fixup
f e3851e8 another fix
```

如果你想组合这些提交(commit) **并重命名这个提交(commit)**, 你应该在第二个提交(commit)旁边添加一个`r`，或者更简单的用`s` 替代 `f`:

```vim
pick a9c8a1d Some refactoring
pick 01b2fd8 New awesome feature
s b729ad5 fixup
s e3851e8 another fix
```

你可以在接下来弹出的文本提示框里重命名提交(commit)。

```vim
Newer, awesomer features

# Please enter the commit message for your changes. Lines starting
# with '#' will be ignored, and an empty message aborts the commit.
# rebase in progress; onto 8074d12
# You are currently editing a commit while rebasing branch 'master' on '8074d12'.
#
# Changes to be committed:
#	modified:   README.md
#

```

如果成功了, 你应该看到类似下面的内容:

```sh
(master)$ Successfully rebased and updated refs/heads/master.
```

###### 安全合并(merging)策略

`--no-commit` 执行合并(merge)但不自动提交, 给用户在做提交前检查和修改的机会。 `no-ff` 会为特性分支(feature branch)的存在过留下证据, 保持项目历史一致。

```sh
(master)$ git merge --no-ff --no-commit my-branch
```

###### 我需要将一个分支合并成一个提交(commit)

```sh
(master)$ git merge --squash my-branch
```

###### 我只想组合(combine)未推的提交(unpushed commit)

有时候，在将数据推向上游之前，你有几个正在进行的工作提交(commit)。这时候不希望把已经推(push)过的组合进来，因为其他人可能已经有提交(commit)引用它们了。

```sh
(master)$ git rebase -i @{u}
```

这会产生一次交互式的rebase(interactive rebase), 只会列出没有推(push)的提交(commit)， 在这个列表时进行reorder/fix/squash 都是安全的。

###### 检查是否分支上的所有提交(commit)都合并(merge)过了

检查一个分支上的所有提交(commit)是否都已经合并(merge)到了其它分支, 你应该在这些分支的head(或任何 commits)之间做一次diff:

```sh
(master)$ git log --graph --left-right --cherry-pick --oneline HEAD...feature/120-on-scroll
```

这会告诉你在一个分支里有而另一个分支没有的所有提交(commit), 和分支之间不共享的提交(commit)的列表。 另一个做法可以是:

```sh
(master)$ git log master ^feature/120-on-scroll --no-merges
```

###### 交互式rebase(interactive rebase)可能出现的问题

###### 这个rebase 编辑屏幕出现'noop'

如果你看到的是这样:

```
noop
```

这意味着你rebase的分支和当前分支在同一个提交(commit)上, 或者 *领先(ahead)* 当前分支。 你可以尝试:

* 检查确保主(master)分支没有问题
* rebase  `HEAD~2` 或者更早

#### 有冲突的情况

如果你不能成功的完成rebase, 你可能必须要解决冲突。

首先执行 `git status` 找出哪些文件有冲突:

```sh
(my-branch)$ git status
On branch my-branch
Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git checkout -- <file>..." to discard changes in working directory)

	modified:   README.md
```

在这个例子里面, `README.md` 有冲突。 打开这个文件找到类似下面的内容:

```vim
   <<<<<<< HEAD
   some code
   =========
   some code
   >>>>>>> new-commit
```

你需要解决新提交的代码(示例里, 从中间`==`线到`new-commit`的地方)与`HEAD` 之间不一样的地方.

有时候这些合并非常复杂，你应该使用可视化的差异编辑器(visual diff editor):

```sh
(master*)$ git mergetool -t opendiff
```

在你解决完所有冲突和测试过后, `git add` 变化了的(changed)文件, 然后用`git rebase --continue` 继续rebase。

```sh
(my-branch)$ git add README.md
(my-branch)$ git rebase --continue
```

如果在解决完所有的冲突过后，得到了与提交前一样的结果, 可以执行`git rebase --skip`。

任何时候你想结束整个rebase 过程，回来rebase前的分支状态, 你可以做:

```sh
(my-branch)$ git rebase --abort
```

##### Stash

###### 暂存所有改动

暂存你工作目录下的所有改动

```sh
$ git stash
```

你可以使用`-u`来排除一些文件

```sh
$ git stash -u
```

###### 暂存指定文件

假设你只想暂存某一个文件

```sh
$ git stash push working-directory-path/filename.ext
```

假设你想暂存多个文件

```sh
$ git stash push working-directory-path/filename1.ext working-directory-path/filename2.ext
```

###### 暂存时记录消息

这样你可以在`list`时看到它

```sh
$ git stash save <message>
```

或

```sh
$ git stash push -m <message>
```

###### 使用某个指定暂存

首先你可以查看你的`stash`记录

```sh
$ git stash list
```

然后你可以`apply`某个`stash`

```sh
$ git stash apply "stash@{n}"
```

此处， 'n'是`stash`在栈中的位置，最上层的`stash`会是0

除此之外，也可以使用时间标记(假如你能记得的话)。

```sh
$ git stash apply "stash@{2.hours.ago}"
```

###### 暂存时保留未暂存的内容

你需要手动create一个`stash commit`， 然后使用`git stash store`。

```sh
$ git stash create
$ git stash store -m "commit-message" CREATED_SHA1
```



##### 跟踪文件(Tracking Files)

###### 我只想改变一个文件名字的大小写，而不修改内容

```sh
(master)$ git mv --force myfile MyFile
```

###### 我想从Git删除一个文件，但保留该文件

```sh
(master)$ git rm --cached log.txt
```
