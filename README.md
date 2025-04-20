# GTVelo
RNA velocity inference based on graph transformer

GTVelo extends [LatentVelo](https://github.com/Spencerfar/LatentVelo)  by introducing a Graph Transformer-based Neural ODE model. This novel approach incorporates multi-head attention mechanisms to overcome single initial state limitations, enabling dynamic learning of global cell-to-cell relationships. The model supports flexible modeling of multi-branch trajectories and integration of multi-omics data.

## Installation
The environment can be set up using either of the following methods:
### Using conda
conda env create -f environment.yml

### Using pip
pip install -r requirements.txt

# Using 
Detailed implementation steps in notebooks.For step-by-step instructions and examples, please refer to the provided  files..ipynb(---)

```
notebook/
├── fig2/
│   ├── data/             
│   │   ├── raw/          # Original experimental data
│   │   └── processed/    # Preprocessed adata and latent_adata objects
├── fig3/
│   └── ...              # Similar structure for other figures
```
# Acknowledgments

This project is based on [LatentVelo](https://github.com/Spencerfar/LatentVelo) by Spencerfar. The original code is licensed under the MIT License.

## License

This project is licensed under the MIT License - see below for details:

MIT License

Copyright (c) 2022 Spencerfar

Copyright (c) [2025] [XJU]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
