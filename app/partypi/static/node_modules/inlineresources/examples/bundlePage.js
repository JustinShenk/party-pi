#!/usr/bin/env node

const path = require('path'),
      puppeteer = require('puppeteer');

const fileUrl = (relPath) => {
    return 'file://' + path.resolve(process.cwd(), relPath);
};

const bundlePage = async (url) => {
    const browser = await puppeteer.launch({args: ['--allow-file-access-from-files']}),
          page = await browser.newPage(),
          pageUrl = fileUrl(url);

    page.on('console', msg => {
        for (let i = 0; i < msg.args().length; ++i) {
            console.warn(`${i}: ${msg.args()[i]}`);
        }
    });
    page.on('pageerror', msg => {
        console.error(msg);
    });

    await page.goto(fileUrl('examples/bundlePage.html'));

    const xhtml = await page.evaluate(function (pageUrl) {
        return bundlePage(pageUrl);
    }, pageUrl);
    console.log(xhtml);

    browser.close();
};

const main = async () => {
    if (process.argv.length !== 3) {
        console.log('Usage: ' + path.basename(process.argv[1]) + ' PAGE_TO_INLINE');
        console.log("Inlines resources of a given page into one big XHTML document");
        process.exit(1);
    }

    const pageToInline = process.argv[2];

    await bundlePage(pageToInline);
};

(async () => {
    try {
        await main();
        process.exit(0);
    } catch (e) {
        console.error(e);
        process.exit(1);
    }
})();
