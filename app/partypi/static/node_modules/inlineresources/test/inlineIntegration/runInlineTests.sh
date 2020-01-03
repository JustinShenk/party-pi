#!/bin/bash
#
# Runs a characterisation test against a list of web pages. This test protects
# the library against unwanted changes to existing behaviour and at the same
# time integrates against web pages in the wild.
#
# Each page is inlined with all its assets and a checksum is calculated for
# the resulting page. This checksum is then matched against a previously
# derived checksum. A mismatch can indicate both an intentional or unwanted
# change in the inlining algorithm.

set -e
set -o pipefail

failedTests=0

testFile() {
    local sourceFile="$1"
    local testReference="$2"
    local inlinedFilePath="build/tests/${testReference}.xhtml"
    local fileTargetSha="test/inlineIntegration/checksums/${testReference}_sha.txt"

    echo "Generating inlined version of ${sourceFile}"
    ./examples/bundlePage.js "$sourceFile" > "$inlinedFilePath"

    echo "Comparing checksum of file with target"
    checksum=$(shasum "$inlinedFilePath" | cut -d' ' -f1)
    if ! echo "$checksum" | diff - "$fileTargetSha" > /dev/null; then
        local expected=$(cat "$fileTargetSha")
        echo "Expected ${expected} but got ${checksum}"
        failedTests=$(expr ${failedTests} + 1)
        echo "FAIL"
    else
        echo "SUCCESS"
    fi
}

takeScreenshot() {
    local testReference="$1"
    local inlinedFilePath="build/tests/${testReference}.xhtml"
    local dateSuffix=$(date +%y%m%d_%H%M%S)
    local screenshotPath="build/tests/${testReference}_${dateSuffix}.png"

    echo "Taking a screenshot of ${inlinedFilePath}, writing to ${screenshotPath}"
    ./test/inlineIntegration/rasterize.js "$inlinedFilePath" "$screenshotPath"
}

downloadPageData() {
    local testReference="$1"
    local targetDirectory="$2"
    local pageDataUrl="http://cburgmer.github.io/inlineresources/testData/${testReference}.tar.bz"

    if [[ ! -d "${targetDirectory}/${testReference}" ]]; then
        echo "Downloading full page from ${pageDataUrl}"
        wget --directory-prefix="$targetDirectory" "$pageDataUrl"
        tar -xjf "${targetDirectory}/${testReference}.tar.bz"
    fi
}

runTest() {
    local testReference="$1"
    local relativeLocalPageUrl="$2"
    local downloadsDir="./build/tests/downloads"

    mkdir -p "$downloadsDir"

    echo "Testing against ${testReference}"
    downloadPageData "${testReference}" "${downloadsDir}"
    testFile "${downloadsDir}/${testReference}/${relativeLocalPageUrl}" "${testReference}"
    takeScreenshot "$testReference"
}

main() {
    runTest "github" "github.com/index.html"
    # Flaky, doesn't always terminate
    #runTest "twitter" "twitter.com/index.html"

    if [[ "$failedTests" -eq 0 ]]; then
        echo "DONE"
        exit 0
    else
        echo "${failedTests} failing tests"
        exit 1
    fi
}

main
