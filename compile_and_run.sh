make bld/gcc/depclean SRC=$3 BOARD=$2
make bld/gcc/dep BOARD=$2
make bld/$1/depclean BOARD=$2 SRC=$3
make bld/$1/dep BOARD=$2
make bld/$1/all BOARD=$2 SRC=$3
mspdebug --allow-fw-update -v 3300 -d /dev/ttyACM$4 tilib "prog bld/$1/$3.out"
echo "TOOL=$1, BOARD=$2, SRC=$3, ACM=$4"
