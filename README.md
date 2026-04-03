# fooSynaptic.github.io

本仓库为 **GitHub Pages 用户站点**：构建产物发布在默认分支（`master`）根目录；**Hexo 源文件**在分支 **`source`** 上维护。

## 线上地址

- **站点首页**：<https://fooSynaptic.github.io>
- **小说《圣殿守护者》目录**：<https://fooSynaptic.github.io/novel/>

## 仓库与分支说明

| 分支 | 用途 |
|------|------|
| **`source`** | Hexo 工程源码：`_config.yml`、主题配置、`source/_posts/` 文章、`source/novel/` 小说页等。日常写作与改主题在此分支提交。 |
| **`master`** | **仅部署用**：由 `hexo deploy` 将 `public/` 静态文件推送到此分支，供 GitHub Pages 访问。一般不要手工改站点 HTML，除非应急；改版请回到 `source` 再生成部署。 |

## 技术栈

- **静态站**：Hexo（Node.js）
- **主题**：NexT
- **数学公式**：MathJax（见 `_config.yml` / 相关插件配置）
- **部署**：`hexo-deployer-git`，远端为 `origin`，发布分支为 `master`

## 本地常用命令

在项目根目录（检出 `source` 分支后）：

```bash
npm install
npx hexo clean && npx hexo generate
npx hexo server
npx hexo deploy
```

部署前需已配置好 Git 对 GitHub 的推送权限（HTTPS/SSH 均可）。

## 内容与结构提示

- **技术博文**：Markdown 放在 `source/_posts/`，构建后文章 URL 形如 `https://fooSynaptic.github.io/:year/:month/:day/:title/`（与 `permalink` 配置一致）。
- **小说连载**：独立页面位于 `source/novel/`，带插图目录 `source/novel/illustrations/`；入口在站点导航 **「小说」** 与上述 `/novel/` 索引页。

## 联系

见站点副标题或 About 页中的邮箱（以线上页面为准）。

---

*本 README 通过 Hexo `skip_render` 原样复制到发布目录，因此在 `master` 根目录可见，便于在 GitHub 仓库页说明站点用途。*
