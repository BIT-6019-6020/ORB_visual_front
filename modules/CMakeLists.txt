

########################################TODO 测试代码用的
add_executable(test  demo_ForTest.cpp)
target_link_libraries( test ${OpenCV_LIBS}
        ${PROJECT_NAME}_ComputeKeyPointsOcTree
        ${PROJECT_NAME}_ORBextractor
        )


########################################TODO 计算图像金字塔
add_executable(Pyramid
        demo_Pyramid.cpp
        )
target_link_libraries( Pyramid ${OpenCV_LIBS}
        )



########################################TODO ORB中的图像特征点提取比较OPENCV的特征点提取
add_executable(ORBextractor
        demo_ORBextractor.cpp
        )
target_link_libraries( ORBextractor ${OpenCV_LIBS}
        lib_${PROJECT_NAME}_ORBextractor
        )

########################################TODO 可视化四叉树特征点匹配的过程
add_executable(ComputeKeyPointsOctTree
        demo_ComputeKeyPointsOctTree.cpp
        )
target_link_libraries( ComputeKeyPointsOctTree ${OpenCV_LIBS}
        lib_${PROJECT_NAME}_ComputeKeyPointsOcTree
        )

########################################TODO 读取配置文件
add_executable(readSettings
        readSettings.cpp
        )
target_link_libraries( readSettings ${OpenCV_LIBS}
        )
