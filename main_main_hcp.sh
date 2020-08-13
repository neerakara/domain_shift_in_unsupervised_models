#!/bin/bash
for S in {1..5}
do

   for p in 'CK/' 'CX/' 'EK/' 'FP/' 'KE/' 'KT/' 'NK/' 'TC/' 'VA/'      
   do
   
      for u in {2..5}
      do
         
         for na in 202         
         do
         
            for n1 in 0 5            
            do
               
               let n2=10-n1
               echo S p u $S $p $u $na $n1 $n2
               qsub /usr/bmicnas01/data-biwi-01/nkarani/projects/domain_shift_unsupervised/code/v2.0/main_hcp.sh $S $p $u $na $n1 $n2
            
            done
            
         done
    
      done
      
   done
   
done

## parser.add_argument('--sli', type = int, default = 3) # 1, 2, 3, 4, 5
## parser.add_argument('--base', default = "EK/") # CK, CX, EK, FP, KE, KT, NK, TC, VA
## parser.add_argument('--usfact', type = float, default = 3) # 2, 3, 4, 5