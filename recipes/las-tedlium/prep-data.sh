#!/bin/bash

# need absolute paths here:
tedliumdir= { set to /some/location/TEDLIUM_release2, the location the TEDLIUM corpus was unzipped to }
datadir= { set to a location with sufficient free space, and adjust the DATA_DIR in config.las-pyramidal.yaml accordingly }

##### convert to .wav #######

for spl in dev test train
do
  for f in `ls $tedliumdir/$spl/sph/*.sph`
  do
    mkdir -p $datadir/wav/$spl
    filename=$(basename "$f" .sph) 
    sox -t sph -b 16 -e signed -r 16000 -c 1 $f $datadir/wav/$spl/$filename.wav
  done
done

#### prepare database and transcripts #####
mkdir $datadir/db/
mkdir $datadir/transcript/

python extract_db.py "$tedliumdir" "$datadir"
cat $datadir/transcript/dev.char | sed "s/ //g" | sed "s/__/ /g" > $datadir/transcript/dev.words
cat $datadir/transcript/test.char | sed "s/ //g" | sed "s/__/ /g" > $datadir/transcript/test.words
cat $datadir/transcript/train.char | sed "s/ //g" | sed "s/__/ /g" > $datadir/transcript/train.words


