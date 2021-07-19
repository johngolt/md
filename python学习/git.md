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

| 指令           | 作用                             |
| -------------- | -------------------------------- |
| `git clone`    | 从git服务器拉取代码              |
| `git config`   | 配置开发者用户名和邮箱           |
| `git branch`   | 创建、重命名、查看、删除项目分支 |
| `git checkout` | 切换分支                         |
| `git status`   | 查看文件变动状态                 |
| `git add`      | 添加文件变动到暂存区             |
| `git commit`   | 提交文件变动到版本库             |
| `git push`     | 将本地的代码改动推送到服务器     |
| `git pull`     | 将服务器上的最新代码拉取到本地   |
| `git log`      | 查看版本提交记录                 |
| `git tag`      | 为项目标记里程碑                 |

  `git status -s` 或 `git status --short`可以得到一种更为紧凑的格式输出。新添加的未跟踪文件前面有 `??` 标记，新添加到暂存区中的文件前面有 `A` 标记，修改过的文件前面有 `M` 标记。 你可能注意到了 `M` 有两个可以出现的位置，出现在右边的 `M` 表示该文件被修改了但是还没放入暂存区，出现在靠左边的 `M` 表示该文件被修改了并放入了暂存区。

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

`.gitignore` 是在项目中的一个文件，通过设置 `.gitignore`的内容告诉 `Git` 哪些文件应该被忽略不需要推送到服务器，通过以上命令可以创建一个 `.gitignore` 文件，并在编辑器中打开文件，每一行代表一个要忽略的文件或目录

```
demo.html
build/
#忽略 demo.html 文件 和 build/ 目录
```

| 指令     | 作用                     |
| -------- | ------------------------ |
| `git mv` | 移动或重命名文件、目录   |
| `git rm` | 从工作区和暂存区移除文件 |
|          |                          |
|          |                          |
|          |                          |
|          |                          |
|          |                          |
|          |                          |
|          |                          |

