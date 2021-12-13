#include <string>
#include <iostream>

#include "lu_entry.hpp"

int main(int argc, char** argv) {

    int ret = 0;
    //Get command line options
    Opts opts;

    parse_opts(&opts, argc, argv);

        ret = lu_test
        <float>(
            opts.m, // Matrix dimensions
            &opts
            );

    return ret;
}