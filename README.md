
<p align="center">
  <a href=""><img alt="logo" src="https://private-user-images.githubusercontent.com/25851824/492778208-2e8896e2-6eb9-4d94-b632-7385ca7c2ffc.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjE3Mzc1NTMsIm5iZiI6MTc2MTczNzI1MywicGF0aCI6Ii8yNTg1MTgyNC80OTI3NzgyMDgtMmU4ODk2ZTItNmViOS00ZDk0LWI2MzItNzM4NWNhN2MyZmZjLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTEwMjklMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUxMDI5VDExMjczM1omWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWFkMDJkNjY2ODk2ZTZhMGYxNDQ2OTVkNTgxOWIzMGFiMjJhZmY1N2FlZTE1MGFkZWYzY2Y4ZmYwMjU3Y2Y4M2EmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.71Z1Wce9Ni1Kii6kPoSH7j6bAk2OwPa8dvDvgUvR0J8" width="50%"></a>
</p>


# What is HookeAI?

[**Docs**](https://bessagroup.github.io/hookeai/)
| [**Installation**](https://bessagroup.github.io/hookeai/rst_doc_files/getting_started/installation.html)
| [**GitHub**](https://github.com/bessagroup/hookeai)
| [**Paper**](https://doi.org/10.1016/j.jmps.2025.106408)

### Summary
**HookeAI** is an open-source Python package built on [PyTorch](https://pytorch.org/) to perform material model updating at the intersection of **computational mechanics**, **machine learning** and **scientific computing**. At its core lies the **Automatically Differentiable Model Updating (ADiMU)** framework, which enables **finding general history-dependent material models** - conventional (physics-based), neural network (data-driven), or hybrid - from different data sources (e.g., strain-stress data, displacement-force data). It also includes numerous computational resources to support **material modeling research**, namely data generation methods, highly-customizable material model architectures, and data analysis and visualization tools.

<p align="center">
  <a href=""><img alt="logo" src="https://private-user-images.githubusercontent.com/25851824/507083361-c4efc04f-5ee9-4ef5-b557-43010b254410.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjE3NDEyMTgsIm5iZiI6MTc2MTc0MDkxOCwicGF0aCI6Ii8yNTg1MTgyNC81MDcwODMzNjEtYzRlZmMwNGYtNWVlOS00ZWY1LWI1NTctNDMwMTBiMjU0NDEwLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTEwMjklMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUxMDI5VDEyMjgzOFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTY2MTAzOGU5N2VlNjgzYWUxOTllOGFlMWNlMzhiMjFmMGYzOTMwMTkwNGE1YTI1Y2RkNDc1ZmFmYzJmY2Q3ZTMmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.tyW82pY2af2bJVKdkS2_QZdUXqeD5mbu8xtbE6za_dQ" width="80%"></a>
</p>


### Statement of need
In an era where a multidisciplinary approach is increasingly essential for scientific and technological progress, **HookeAI** is a computational platform aiming to integrate three key domains: (1) **computational solid mechanics**, accounting for decades of well-grounded physics-based material modeling, (2) **machine learning**, where deep learning architectures are capable of learning complex material behavior from data, and (3) **scientific computing software**, leveraging modern automatic differentiation, optimization techniques and GPU vectorization to efficiently solve challenging inverse problems. By combining these three pillars in a unique highly-customizable and well-documented software platform, HookeAI addresses the needs and challenges in teaching, research and industry, where **accurate material modeling** is crucial for the design and analysis of engineering systems.

In a **teaching environment**, HookeAI extensive documentation, pre-built material models, user-friendly interfaces, and data visualization tools make it an excellent resource for advanced courses in computational solid mechanics with machine learning, namely to enable hands-on learning and experience. In a **research setting**, HookeAI provides a comprehensive end-to-end platform for developing and testing new material model architectures, setting up reproducible benchmarks, comparing the performance of optimization algorithms, conducting rigorous analyses of material data sets, and much more. Its extensive out-of-the-box functionalities, combined with a modular and highly-customizable design, greatly reduce the time and effort needed to implement new ideas, allowing researchers to focus on scientific innovation. In an **industrial context**, HookeAI enables the discovery of a wide range of material models from numerical or experimental data, providing practical solutions to real-world material modeling challenges. Applications include designing novel material systems with disruptive properties, tailoring material performance for specific uses, improving the accuracy of product design and analysis simulations, supporting multi-objective topology optimization, and beyond.

Consider **leaving a star** :star: if you think HookeAI is useful for the community!

<p align="center">
  <a href=""><img alt="logo" src="https://private-user-images.githubusercontent.com/25851824/507056571-42659c1b-5abf-4aa2-95ff-c67a203527cd.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjE3Mzc2ODcsIm5iZiI6MTc2MTczNzM4NywicGF0aCI6Ii8yNTg1MTgyNC81MDcwNTY1NzEtNDI2NTljMWItNWFiZi00YWEyLTk1ZmYtYzY3YTIwMzUyN2NkLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTEwMjklMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUxMDI5VDExMjk0N1omWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWY4MjRkNTc3NjZmYjQwZjZiMDRmYzcwNTQ2NzlkMjQ3YzU3NTI1NzQ5YjE5MWI2NjM5MGI3YjVlZTMzYjQ1YzcmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.B35hMeK3vDihu5AM1Z9633ttEHZaU5hA5spvdhp3u3M" width="90%"></a>
</p>


### Authorship & Citation
HookeAI was originally developed by Bernardo P. Ferreira<sup>[1](#f1)</sup>.

<sup id="f1"> 1 </sup> Profile: [LinkedIn](https://www.linkedin.com/in/bpferreira/), [ORCID](https://orcid.org/0000-0001-5956-3877), [ResearchGate](https://www.researchgate.net/profile/Bernardo-Ferreira-11?ev=hdr_xprf)


If you use HookeAI in your research or in a scientific publication, please cite:

**Journal of the Mechanics and Physics of Solids** ([paper](https://doi.org/10.1016/j.jmps.2025.106408)):
```
@article{ferreira2025a,
  title = {Automatically Differentiable Model Updating (ADiMU): Conventional, hybrid, and neural network material model discovery including history-dependency},
  journal = {Journal of the Mechanics and Physics of Solids},
  volume = {206},
  pages = {106408},
  year = {2026},
  issn = {0022-5096},
  doi = {https://doi.org/10.1016/j.jmps.2025.106408},
  url = {https://www.sciencedirect.com/science/article/pii/S0022509625003825},
  author = {Bernardo P. Ferreira and Miguel A. Bessa},
  keywords = {Material model, Model updating, Automatic differentiation, History-dependency, Recurrent neural network, Hybrid material model, ADiMU, Open-source}
}
```

----

# Getting started

You can find everything you need to know in [HookeAI documentation](https://bessagroup.github.io/hookeai/)!


----

# Community Support

If you find any **issues**, **bugs** or **problems** with HookeAI, please use the [GitHub issue tracker](https://github.com/bessagroup/hookeai/issues) to report them. Provide a clear description of the problem, as well as a complete report on the underlying details, so that it can be easily reproduced and (hopefully) fixed!

You are also welcome to post there any **questions**, **comments** or **suggestions** for improvement in the [GitHub discussions](https://github.com/bessagroup/hookeai/discussions) space!

Please refer to HookeAI's [Code of Conduct](https://github.com/bessagroup/hookeai/blob/main/CODE_OF_CONDUCT.md).


# Credits

* No contributors yet!


# License

Copyright 2025, Bernardo Ferreira

All rights reserved.

HookeAI is a free and open-source software published under a MIT License.
