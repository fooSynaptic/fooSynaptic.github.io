'use strict';

/**
 * 文中图片使用 ../images/ 以便在 source/_posts 下打开 Markdown 时，
 * 编辑器预览能解析到 source/images/。
 * 生成站点时还原为 /images/，与 GitHub Pages 根路径一致。
 */
hexo.extend.filter.register('after_post_render', function (data) {
  if (!data.content) return data;
  data.content = data.content.replace(/src="\.\.\/images\//g, 'src="/images/');
  data.content = data.content.replace(/src='\.\.\/images\//g, "src='/images/");
  return data;
});
