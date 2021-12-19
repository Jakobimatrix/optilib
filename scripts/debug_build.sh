cd ../build
cmake  .. -DTEST_SETTINGS=True -DCMAKE_BUILD_TYPE=Debug
make -j 8
ctest --output-on-failure

