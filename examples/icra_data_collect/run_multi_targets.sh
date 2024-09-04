#!/bin/bash

## Loop over all povray files and produce images
for i in {59..63} # {55..58} # {51..54} # {47..50} # {43..46} # {36..42} # {29..35} # {23..28} # {16..22} # {9..15} # {2..8}
do
	echo "Processing " $i "th file..."
	python run_simulation.py $i
	python plot_frames.py $i
done

: <<'END_COMMENT'