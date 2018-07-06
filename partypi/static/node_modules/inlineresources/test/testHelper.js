"use strict";

var isPhantomJs = navigator.userAgent.indexOf("PhantomJS") >= 0,
    isRunFromTheProjectRoot = isPhantomJs;

var testDisabledOnCondition = function (condition, text, functionHandle) {
    var spec = it(text, functionHandle);
    if (condition) {
        spec.pend('disabled on this platform');
        console.info('"' + text + '" is disabled on this platform');
    }
    return spec;
};

exports.isChrome = navigator.userAgent.indexOf("Chrom") >= 0;
exports.ifNotInChromeIt = function(text, functionHandle) {
    testDisabledOnCondition(exports.isChrome, text, functionHandle);
};

exports.fixturesPath = (isRunFromTheProjectRoot ? 'test/' : '' ) + 'fixtures/';


exports.loadHTMLDocumentFixture = function (url) {
    return new Promise(function (resolve) {
        var fixtureUrl = exports.fixturesPath + url,
            xhr = new window.XMLHttpRequest();

        xhr.onload = function() {
            resolve(xhr.responseXML);
        };

        xhr.open("GET", fixtureUrl);
        xhr.responseType = "document";
        // Force html https://bugzilla.mozilla.org/show_bug.cgi?id=942138
        xhr.overrideMimeType("text/html");
        xhr.send();
    });
};

exports.readDocumentFixture = function (url) {
    var fixtureUrl = exports.fixturesPath + url,
        xhr = new window.XMLHttpRequest();

    xhr.open('GET', fixtureUrl, false);
    xhr.overrideMimeType('text/xml');
    xhr.send(null);
    return xhr.responseXML;
};

exports.loadHTMLDocumentFixtureWithoutBaseURI = function (url) {
    return exports.loadHTMLDocumentFixture(url).then(function (doc) {
        var baseURILessDoc = document.implementation.createHTMLDocument("");
        baseURILessDoc.documentElement.innerHTML = doc.documentElement.innerHTML;
        return baseURILessDoc;
    });
};

exports.addStyleToDocument = function (doc, styleContent) {
    var styleNode = doc.createElement("style");

    styleNode.type = "text/css";
    styleNode.appendChild(doc.createTextNode(styleContent));

    doc.getElementsByTagName('head')[0].appendChild(styleNode);
};
