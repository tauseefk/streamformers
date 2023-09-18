# streamformers

Rustformers API is synchronous and doesn't directly support streaming responses. As it's quite hard to build anything usable on top of the callback API, I wrote a little example to get streaming response out of rustformers. The solution leverages `std::sync::mpsc` channels. This also works with `actix` however only producers can be cloned so the consumer (inference response sender) still needs to be static.

![streamformers@2x](https://github.com/tauseefk/streamformers/assets/11029896/876368b8-3dde-48c7-b478-8144579a8c34)
