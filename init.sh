#! /bin/bash

if [ $@ = '-h' ] || [ -z $@  ]; then
    echo "Use this script to get all Diving3D necessary files, including the data cubes and STARLIGHT."
    echo "This script will also compile STARLIGHT locally using gfortran."
    echo " "
    echo "Just run this script with your minerva username as an argument when you're ready."

else

    DIVINGDATAPATH=`pwd`/data
    echo $DIVINGDATAPATH
    echo "Getting data cubes..."
    rsync -avz $@@minerva.ufsc.br:/net/califa/natalia/data/diving3d/{cubes,cubes_obs} $DIVINGDATAPATH

    echo "Done!"

    echo "Now let's compile STARLIGHT"
    echo "cd ./starlight"
    cd $DIVINGDATAPATH/starlight
    echo `pwd`

    echo "gfortran PANcMEx_StarlightChains_v03b.for -o PANcMEx_StarlightChains_v03b.exe"
    echo "Compiling STARLIGHT..."

    gfortran PANcMEx_StarlightChains_v03b.for -o PANcMEx_StarlightChains_v03b.exe

    echo "Done!"
    chmod a+x PANcMEx_StarlightChains_v03b.exe

fi
