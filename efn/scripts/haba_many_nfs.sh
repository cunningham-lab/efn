for ds in {2..9}
do
  for rs in {0..9}
  do
    sbatch train_nf_helper.sh $ds $rs
  done
done

