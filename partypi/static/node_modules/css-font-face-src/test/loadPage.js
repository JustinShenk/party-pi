/* jshint phantom: true */
var system = require('system');
var fs = require('fs');
var page = require('webpage').create();
page.onConsoleMessage = function(msg) {
    console.log(msg);
};
var path = system.args[1];
if (path.indexOf('://') < 0) {
    if (!fs.isAbsolute(path)) {
        path = fs.workingDirectory + '/' + path;
    }
    path = 'file://' + path;
}
page.open(path, function () {
    phantom.exit();
});
