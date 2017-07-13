# Only work in Unix based system

input=$1
perl=`which perl`

$perl -pi -e 's/\s*$/\n/' $input

