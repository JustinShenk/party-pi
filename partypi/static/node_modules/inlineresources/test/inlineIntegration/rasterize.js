#!/usr/bin/env node
/* jshint ignore:start */
const path = require('path'),
      puppeteer = require('puppeteer');

const fileUrl = (relPath) => {
    return 'file://' + path.resolve(process.cwd(), relPath);
};

const takeScreenshot = async (url, targetPath) => {
    const browser = await puppeteer.launch({args: ['--allow-file-access-from-files']}),
          page = await browser.newPage(),
          pageUrl = fileUrl(url);

    await page.goto(pageUrl);
    await page.reload(); // Work around parse error `error on line 36 at column 187413: Char 0x0 out of allowed range`
    await page.reload();
    await page.reload();

    await page.screenshot({path: targetPath, fullPage: true});
    browser.close();
};

const main = async () => {
    if (process.argv.length !== 4) {
        console.log('Usage: ' + path.basename(process.argv[1]) + ' URL TARGET_PATH');
        console.log("Takes a screenshot of a given URL");
        process.exit(1);
    }

    const url = process.argv[2],
          target = process.argv[3];

    await takeScreenshot(url, target);
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
/* jshint ignore:end */
