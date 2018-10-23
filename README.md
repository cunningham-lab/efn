# efn
Trains exponential family networks (EFNs) to approximate exponential family models.

Installation: <br/>
The efn git repository relies on a second git repo (tf_util).  tf_util is installed</br>
as a linked dependency when running the setup.py script.</br>
<code>python3 setup.py install </code>

To homogenize development environments, use a virtual environment with the
dependencies listed in requirements.txt.  For example, create an anaconda
environment </br>
<code>conda create -n efn</code> </br>
<code>source activate efn</code> </br>
<code>conda install --yes --file requirements.txt</code> </br>

Usage example:
Learn the 3-dimensional hierarchical dirichlet posterior inference model.  This will
train the same EFN visualized in Fig. 1 of (link to arxiv post). </br>
<code>cd scripts/</code> </br>
<code>python3 train_efn_helper.py HierarchicalDirichlet 3 0 0 example</code> </br>
