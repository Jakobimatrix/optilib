cd ../build
cmake  .. -DTEST_SETTINGS=True -DCMAKE_BUILD_TYPE=Release
make -j 8
ctest --output-on-failure
