cmake_minimum_required(VERSION 3.0)

project("test_opti_lib" VERSION 1.0
             DESCRIPTION "Boost Tests for opti_lib"
             HOMEPAGE_URL "https://github.com/Jakobimatrix/optilib")

set(CMAKE_VERBOSE_MAKEFILE on)

set(BOOST_INCLUDE_DIRS $boost_installation_prefix/include)

include_directories(${Boost_INCLUDE_DIRS})
link_directories (${Boost_LIBRARY_DIRS})

find_package(Boost COMPONENTS system unit_test_framework filesystem)

if( Boost_FOUND )

  add_executable(test_opti_lib_stopping test_stoppingconditions.cpp)
  add_executable(test_opti_lib_sectioning test_sectioning.cpp)

  list(APPEND TARGET_LIBS
      PRIVATE opti_lib
      PRIVATE ${Boost_FILESYSTEM_LIBRARY}
      PRIVATE ${Boost_SYSTEM_LIBRARY}
      PRIVATE ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})

  target_link_libraries(test_opti_lib_stopping
                        ${TARGET_LIBS}
                        )
  target_link_libraries(test_opti_lib_sectioning
                        ${TARGET_LIBS}
                        )

  # declares a test with executable
  add_test(NAME test_stopping_conditions COMMAND test_opti_lib_stopping)
  add_test(NAME test_sectioning COMMAND test_opti_lib_sectioning)
else()
  MESSAGE("BOOST NOT FOUND!! CAN NOT RUN TESTS.")
endif()
