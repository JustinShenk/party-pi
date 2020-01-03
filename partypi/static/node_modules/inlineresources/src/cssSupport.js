"use strict";

exports.unquoteString = function (quotedUrl) {
    var doubleQuoteRegex = /^"(.*)"$/,
        singleQuoteRegex = /^'(.*)'$/;

    if (doubleQuoteRegex.test(quotedUrl)) {
        return quotedUrl.replace(doubleQuoteRegex, "$1");
    } else {
        if (singleQuoteRegex.test(quotedUrl)) {
            return quotedUrl.replace(singleQuoteRegex, "$1");
        } else {
            return quotedUrl;
        }
    }
};

exports.rulesForCssText = function (styleContent) {
    var doc = document.implementation.createHTMLDocument(""),
        styleElement = document.createElement("style"),
        rules;

    styleElement.textContent = styleContent;
    // the style will only be parsed once it is added to a document
    doc.body.appendChild(styleElement);
    rules = styleElement.sheet.cssRules;

    return Array.prototype.slice.call(rules);
};

exports.cssRulesToText = function (cssRules) {
    return cssRules.reduce(function (cssText, rule) {
        return cssText + rule.cssText;
    }, '');
};

exports.exchangeRule = function (cssRules, rule, newRuleText) {
    var ruleIdx = cssRules.indexOf(rule);

    // We create a new document and stylesheet to parse the rule,
    // instead of relying on rule.parentStyleSheet, because
    // rule.parentStyleSheet may be null
    // (https://github.com/cburgmer/inlineresources/issues/3)
    cssRules[ruleIdx] = exports.rulesForCssText(newRuleText)[0];
};

// Workaround for https://bugzilla.mozilla.org/show_bug.cgi?id=443978
exports.changeFontFaceRuleSrc = function (cssRules, rule, newSrc) {
    var newRuleText = '@font-face { font-family: ' + rule.style.getPropertyValue("font-family") + '; ';

    if (rule.style.getPropertyValue("font-style")) {
        newRuleText += 'font-style: ' + rule.style.getPropertyValue("font-style") + '; ';
    }

    if (rule.style.getPropertyValue("font-weight")) {
        newRuleText += 'font-weight: ' + rule.style.getPropertyValue("font-weight") + '; ';
    }

    newRuleText += 'src: ' + newSrc + '}';
    exports.exchangeRule(cssRules, rule, newRuleText);
};
