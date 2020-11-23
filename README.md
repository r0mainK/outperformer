# Outperformer

Repository containing the implementations related to my blog post series on scaling Transformers:

- [Scaling Transformers: Reform your ways](https://keramitas.io/2020/11/24/scaling-transformers-reform-your-ways.html)
- [Scaling Transformers: Perform at your best]()

Check those out for a detailed explanation of the code and additional information (like reference papers and related codebases).

For now, the codebase is split between 3 files:

- implementation of fast attention in the [fast_attention.py](src/fast_attention.py) file
- implementation of reversible layers in the [reversible.py](src/reversible.py) file
- implementation of a headless Reformer + Performer model (a BERT-like MLM with the above modifications) in the [performer.py](src/performer.py) file


If you have any questions (and couldn't find an answer in the post), feel free to open an issue !

Regarding contributions, bug reports (and fixes) are greatly appreciated - although I hope there won't be any :p I don't know yet in which direction this repository will go, whether it will stay as is or incorporate additional features, so if you have ideas please open an issue to talk about them ! Any new feature should be in the spirit of the existing code: aiming at scaling Transformer MLMs through architectural innovations.

If you end up contributing, please review the [guidelines](CONTRIBUTING.md) first.

All of this is released under the [MIT License](LICENSE) so feel free to use it as you wish :D
