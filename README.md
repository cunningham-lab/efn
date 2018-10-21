# efn
Trains exponential family networks (EFNs) to approximate exponential family models.

Installation: <br/>
The efn git repository relies on a second git repo (tf_util) as a submodule.  When
cloning the efn repo, use the --recurse option to clone the submodule as well. </br>
<code>git clone --recurse https://github.com/cunningham-lab/efn.git </code>

Then update the tf_util submodule files to the most recent version. </br>
<code>git submodule foreach git pull origin master </code>
