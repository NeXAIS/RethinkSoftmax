#!/bin/sh

source activate zzh_env110
cd src

python main_incremental.py --approach lwf --num-tasks 10 --datasets 'cifar100' 
python main_incremental.py --approach lwf-br --num-tasks 10 --datasets 'cifar100' 
python main_incremental.py --approach lwf-s --num-tasks 10 --datasets 'cifar100' 
python main_incremental.py --approach lwf-sr --num-tasks 10 --datasets 'cifar100' 
python main_incremental.py --approach _lwf --num-tasks 10 --datasets 'cifar100' 
python main_incremental.py --approach _lwf-br --num-tasks 10 --datasets 'cifar100' 
python main_incremental.py --approach _lwf-s --num-tasks 10 --datasets 'cifar100' 
python main_incremental.py --approach _lwf-sr --num-tasks 10 --datasets 'cifar100' 

python main_incremental.py --approach lucir-sr --num-tasks 10 --datasets 'cifar100' --gamma1  0.2
python main_incremental.py --approach lucir-sr --num-tasks 10 --datasets 'cifar100' --gamma1  0.3
python main_incremental.py --approach lucir-sr --num-tasks 10 --datasets 'cifar100' --gamma1  0.4
python main_incremental.py --approach lucir-sr --num-tasks 10 --datasets 'cifar100' --gamma1  0.5
python main_incremental.py --approach lucir-sr --num-tasks 10 --datasets 'cifar100' --gamma1  0.6
python main_incremental.py --approach lucir-sr --num-tasks 10 --datasets 'cifar100' --gamma1  0.7
python main_incremental.py --approach lucir-sr --num-tasks 10 --datasets 'cifar100' --gamma1  0.8
python main_incremental.py --approach lucir-sr --num-tasks 10 --datasets 'cifar100' --gamma1  0.9
python main_incremental.py --approach lucir-sr --num-tasks 10 --datasets 'cifar100' --gamma1  1.0

python main_incremental.py --approach lucir-s --num-tasks 10 --datasets 'cifar100' --gamma1  0.2
python main_incremental.py --approach lucir-s --num-tasks 10 --datasets 'cifar100' --gamma1  0.3
python main_incremental.py --approach lucir-s --num-tasks 10 --datasets 'cifar100' --gamma1  0.4
python main_incremental.py --approach lucir-s --num-tasks 10 --datasets 'cifar100' --gamma1  0.5
python main_incremental.py --approach lucir-s --num-tasks 10 --datasets 'cifar100' --gamma1  0.6
python main_incremental.py --approach lucir-s --num-tasks 10 --datasets 'cifar100' --gamma1  0.7
python main_incremental.py --approach lucir-s --num-tasks 10 --datasets 'cifar100' --gamma1  0.8
python main_incremental.py --approach lucir-s --num-tasks 10 --datasets 'cifar100' --gamma1  0.9
python main_incremental.py --approach lucir-s --num-tasks 10 --datasets 'cifar100' --gamma1  1.0





