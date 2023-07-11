# Are you ready to write high quality code?

The idea of tinygrad is to build a <5000 line library capable of training a wide variety of ML models at 80%+ max theoretical speed across a huge variety of hardware.

There is almost no boilerplate code anywhere in this library, and you should help keep it that way. If the code you are contributing to core tinygrad, in `tinygrad/`, isn't some of the highest quality code you've written in your life, either put in the effort to make it great, or don't bother. (other directories have a slightly more relaxed standard)

There is a linter, but it's not complete. Spend a little time reading the existing code to get a feel for the style.

I love PRs where I can look at them and just say, yes, this will improve the codebase and click merge. If you have an incomplete PR, feel free to post it as a draft.

As my operating systems professor taught me, code is written to be read by humans. We value readability over performance and line count, but low line count is often a good proxy for readability. However, any PRs that look like code golf will immediately be closed.

There are a few basic ways to contribute:

## Bug-fixes

These are the most straightforward. Discover a bug. Add a test to reproduce it. Write a clean fix. Submit a PR. Confirm CI passes.

## Conceptual Cleanups

This is some of the highest value work in tinygrad. If you realize two 50 line functions are basically the same thing, and you can merge them, amazing! Things that look confusing and are hard to follow are probably poorly written. If you can rewrite the code and be like, oh that's a ton simpler, by all means do so. Make sure you have good test coverage around what you are changing.

## Better Testing

Always welcome! Think about how robust and fast your tests are though. How likely is this test to catch a bug? Tests that run in CI go in `test/`, except for the ones in `test/external/`. We have a few things like fuzzers in there.

## Speed improvements

tinygrad is a JIT compiler, so speed improvements refer to both compile-time and runtime. Speed improvements to the python based compiler are welcome, but please include benchmarks and good tests around the things that you are changing. If you are sacrificing readability for speed, don't bother. Speed improvements to the generated code usually come from conceptual cleanups. Generated code improvements are probably the hardest thing to work on in tinygrad, since they must be done in a very generic way.

## Features

This is a trickier one. If there is a feature in PyTorch and numpy that you have actually seen people use, we probably want it. All new features must include good robust tests, and in general, matching the PyTorch API is good.
