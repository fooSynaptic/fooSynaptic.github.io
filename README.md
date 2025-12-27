# fooSynaptic 博客源码

这是使用 Hexo + NexT 主题构建的博客源码。

## 环境要求

- Node.js 18+
- npm 或 yarn

## 快速开始

### 安装依赖

```bash
npm install
```

### 本地预览

```bash
npx hexo server
```

然后访问 http://localhost:4000

### 创建新文章

```bash
npx hexo new "文章标题"
```

新文章将在 `source/_posts/` 目录下创建。

### 生成静态文件

```bash
npx hexo generate
```

生成的文件在 `public/` 目录。

### 清理缓存

```bash
npx hexo clean
```

## 部署到 GitHub Pages

### 方法一：手动部署

1. 生成静态文件：`npx hexo generate`
2. 将 `public/` 目录下的所有文件复制到 GitHub Pages 仓库根目录
3. 提交并推送到 GitHub

### 方法二：使用 hexo-deployer-git（推荐）

已配置好 `hexo-deployer-git`，直接运行：

```bash
npx hexo deploy
```

> 注意：首次部署可能需要配置 Git 认证。

## 目录结构

```
hexo-source/
├── _config.yml          # Hexo 主配置文件
├── _config.next.yml     # NexT 主题配置文件
├── source/
│   ├── _posts/          # 博客文章 (Markdown)
│   ├── images/          # 图片资源
│   ├── tags/            # 标签页面
│   └── categories/      # 分类页面
├── themes/              # 主题目录
├── public/              # 生成的静态文件
└── node_modules/        # 依赖包
```

## 文章格式

每篇文章开头需要包含 front matter：

```yaml
---
title: "文章标题"
date: 2024-01-01 12:00:00
tags:
  - 标签1
  - 标签2
categories:
  - 分类1
---

文章内容...
```

## 主题定制

NexT 主题配置在 `_config.next.yml` 文件中，可以修改：

- 配色方案 (`scheme`)
- 导航菜单 (`menu`)
- 侧边栏 (`sidebar`)
- 头像 (`avatar`)
- 社交链接 (`social`)
- 代码高亮 (`codeblock`)

## 常用命令

| 命令 | 说明 |
|------|------|
| `npx hexo new "title"` | 创建新文章 |
| `npx hexo new page "name"` | 创建新页面 |
| `npx hexo generate` | 生成静态文件 |
| `npx hexo server` | 启动本地服务器 |
| `npx hexo deploy` | 部署到远程 |
| `npx hexo clean` | 清理缓存和生成文件 |

## 分支说明

- `master` 分支：存放生成的静态网站文件（用于 GitHub Pages 托管）
- `source` 分支：存放 Hexo 源码（本分支）

