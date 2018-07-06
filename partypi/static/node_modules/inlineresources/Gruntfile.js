/*global module:false*/
"use strict";

module.exports = function (grunt) {
    grunt.initConfig({
        pkg: grunt.file.readJSON('package.json'),
        karma: {
            options: {
                files: [
                    'build/testSuite.js',
                    {pattern: 'test/fixtures/**', included: false}
                ],
                frameworks: ['jasmine'],
                reporters: 'dots'
            },
            ci: {
                proxies: {
                    '/fixtures/': 'http://localhost:9988/base/test/fixtures/'
                },
                port: 9988,
                singleRun: true,
                browsers: ['ChromeHeadless']
            },
            local: {
                proxies: {
                    '/fixtures/': 'http://localhost:9989/base/test/fixtures/'
                },
                port: 9989,
                background: true,
                singleRun: false
            }
        },
        exec: {
            // work around https://github.com/substack/node-browserify/pull/1151
            bundle: './node_modules/.bin/browserify --standalone <%= pkg.name %> --external url --external css-font-face-src src/inline.js | ./node_modules/.bin/derequire > build/<%= pkg.name %>.bundled.js'
        },
        browserify: {
            url: {
                src: [],
                dest: 'build/dependencies/url.js',
                options: {
                    require: ['url'],
                    browserifyOptions: {
                        standalone: 'url'
                    }
                }
            },
            testSuite: {
                src: 'test/specs/*.js',
                dest: 'build/testSuite.js',
                options: {
                    browserifyOptions: {
                        debug: true
                    }
                }
            },
            allinone: {
                src: 'src/inline.js',
                dest: 'build/<%= pkg.name %>.allinone.js',
                options: {
                    browserifyOptions: {
                        standalone: '<%= pkg.name %>'
                    }
                }
            }
        },
        clean: {
            dist: ['build/*.js', 'build/dependencies/', 'dist/'],
            all: ['build']
        },
        concat: {
            dist: {
                options: {
                    // Work around https://github.com/substack/node-browserify/issues/670
                    banner: '/*! <%= pkg.name %> - v<%= pkg.version %> - ' +
                        '<%= grunt.template.today("yyyy-mm-dd") %>\n' +
                        '* <%= pkg.homepage %>\n' +
                        '* Copyright (c) <%= grunt.template.today("yyyy") %> <%= pkg.author.name %>;' +
                        ' Licensed <%= pkg.license %> */\n' +
                        ["// UMD header",
                         "(function (root, factory) {",
                         "    if (typeof define === 'function' && define.amd) {",
                         "        define(['url', 'css-font-face-src'], function (a0,b1) {",
                         "            return (root['<%= pkg.name %>'] = factory(a0,b1));",
                         "        });",
                         "    } else if (typeof exports === 'object') { // browserify context",
                         "        var f = factory(require('url'), require('css-font-face-src'));",
                         "        for(var prop in f) exports[prop] = f[prop];",
                         "    } else {",
                         "        root['<%= pkg.name %>'] = factory(url,cssFontFaceSrc);",
                         "    }",
                         "}(this, function (url, cssFontFaceSrc) {",
                         "    var modules = {url: url, 'css-font-face-src': cssFontFaceSrc};",
                         "    var require = function (name) { if (modules[name]) { return modules[name]; } else { throw new Error('Module not found: ' + name); }; };",
                         "    // cheat browserify module to leave the function reference for us",
                         "    var module = {}, exports={};",
                         "    // from here on it's browserify all the way\n"].join('\n'),
                    footer: ["\n    // back to us",
                             "    return module.exports;",
                             "}));\n"].join('\n')
                },
                src: ['build/<%= pkg.name %>.bundled.js'],
                dest: 'dist/<%= pkg.name %>.js'
            }
        },
        uglify: {
            allinone: {
                options: {
                    banner:'/*! <%= pkg.name %> - v<%= pkg.version %> - ' +
                        '<%= grunt.template.today("yyyy-mm-dd") %>\n' +
                        '* <%= pkg.homepage %>\n' +
                        '* Copyright (c) <%= grunt.template.today("yyyy") %> <%= pkg.author.name %>;' +
                        ' Licensed <%= pkg.license %> */\n' +
                        '/* Integrated dependencies:\n' +
                        ' * url (MIT License),\n' +
                        ' * css-font-face-src (BSD License) */\n'
                },
                files: {
                    'dist/<%= pkg.name %>.allinone.js': ['build/<%= pkg.name %>.allinone.js']
                }
            }
        },
        watch: {
            karma: {
                files: [
                    'src/*.js',
                    'test/specs/*.js',
                    // Ignore files generated by flycheck
                    '!**/flycheck_*.js'
                ],
                tasks: ['browserify:testSuite', 'karma:local:run']
            },
            karmaci: {
                files: [
                    'src/*.js',
                    'test/specs/*.js',
                    // Ignore files generated by flycheck
                    '!**/flycheck_*.js'
                ],
                tasks: ['browserify:testSuite', 'karma:ci']
            },
            build: {
                files: [
                    'src/*.js',
                    'test/specs/*.js'
                ],
                tasks: ['browserify:testSuite']
            }
        },
        jshint: {
            options: {
                curly: true,
                eqeqeq: true,
                immed: true,
                latedef: true,
                newcap: true,
                noarg: true,
                undef: true,
                unused: true,
                eqnull: true,
                trailing: true,
                browser: true,
                node: true,
                strict: true,
                globals: {
                    Promise: true,
                    require: true,
                    exports: true,

                    url: true
                },
                exported: ['inline', 'inlineCss', 'inlineUtil']
            },
            uses_defaults: [
                'src/*.js',
                'Gruntfile.js',
            ],
            with_overrides: {
                options: {
                    globals: {
                        Promise: true,
                        jasmine: true,
                        describe: true,
                        it: true,
                        xit: true,
                        beforeEach: true,
                        afterEach: true,
                        expect: true,
                        spyOn: true,

                        url: true
                    },
                    ignores: ['test/fixtures/']
                },
                files: {
                    src: ['test/']
                }
            }
        },
        "regex-check": {
            files: [
                'src/*',
                // 'test/{,*/}*'
                'test/*.html',
                'test/*.js',
                'test/*/*.html',
            ],
            options: {
                pattern : /FIXME/g
            }
        }
    });

    grunt.loadNpmTasks('grunt-contrib-concat');
    grunt.loadNpmTasks('grunt-contrib-jshint');
    grunt.loadNpmTasks('grunt-contrib-uglify');
    grunt.loadNpmTasks('grunt-regex-check');
    grunt.loadNpmTasks('grunt-contrib-watch');
    grunt.loadNpmTasks('grunt-contrib-clean');
    grunt.loadNpmTasks('grunt-browserify');
    grunt.loadNpmTasks('grunt-umd');
    grunt.loadNpmTasks('grunt-karma');
    grunt.loadNpmTasks('grunt-exec');

    grunt.registerTask('testDeps', [
        'browserify:url'
    ]);

    grunt.registerTask('testWatch', [
        'karma:local:start',
        'watch:karma'
    ]);

    grunt.registerTask('testWatchCi', [
        'karma:local:start',
        'watch:karmaci'
    ]);

    grunt.registerTask('test', [
        'browserify:testSuite',
        'jshint',
        'karma:ci',
        'regex-check'
    ]);

    grunt.registerTask('build', [
        'exec:bundle',
        'concat:dist',
        'browserify:allinone',
        'uglify'
    ]);

    grunt.registerTask('default', [
        'clean:dist',
        'testDeps',
        'test',
        'build'
    ]);

};
