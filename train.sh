#!/bin/sh

domains=('books' 'dvd' 'electronics' 'kitchen' 'video')
for src_domain in ${domains[@]};
do
	for tar_domain in  ${domains[@]};
	do
		if [ $src_domain != $tar_domain ];
		then
			
			python train_cnn_aux.py -s $src_domain -t $tar_domain
		fi
	done
done
          