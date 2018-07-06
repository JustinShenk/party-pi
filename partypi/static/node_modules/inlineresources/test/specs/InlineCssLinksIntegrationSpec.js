"use strict";

var inline = require('../../src/inline'),
    inlineCss = require('../../src/inlineCss'),
    util = require('../../src/util');


describe("Inline CSS content (integration)", function () {
    var doc, ajaxSpyUrlMap = {};

    beforeEach(function () {
        doc = document.implementation.createHTMLDocument("");

        spyOn(util, "ajax").and.callFake(function (url) {
            var respondWith = ajaxSpyUrlMap[url];
            if (respondWith) {
                return Promise.resolve(respondWith);
            }
        });
    });

    var appendStylesheetLink = function (doc, href) {
        var cssLink = window.document.createElement("link");
        cssLink.href = href;
        cssLink.rel = "stylesheet";
        cssLink.type = "text/css";

        doc.head.appendChild(cssLink);
    };

    var mockAjaxWithSuccess = function (params) {
        ajaxSpyUrlMap[params.url] = params.respondWith;
    };

    // https://github.com/cburgmer/rasterizeHTML.js/issues/42
    it("should correctly inline a font as second rule with CSSOM fallback", function (done) {
        mockAjaxWithSuccess({
            url: "some.css",
            respondWith: 'p { font-size: 14px; } @font-face { font-family: "test font"; src: url("fake.woff"); }'
        });
        mockAjaxWithSuccess({
            url: "fake.woff",
            respondWith: "this is not a font"
        });

        appendStylesheetLink(doc, "some.css");

        inline.loadAndInlineCssLinks(doc, {}).then(function () {
            expect(doc.head.getElementsByTagName("style")[0].textContent).toMatch(
                /p\s*\{\s*font-size:\s*14px;\s*\}\s*@font-face\s*\{\s*font-family:\s*["']test font["'];\s*src:\s*url\("?data:font\/woff;base64,dGhpcyBpcyBub3QgYSBmb250"?\);\s*\}/
            );

            done();
        });
    });
});
