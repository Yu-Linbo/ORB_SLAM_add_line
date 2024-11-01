/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef MAP_H
#define MAP_H

#include "MapPoint.h"
#include "MapLine.h"
#include "KeyFrame.h"
#include <set>

#include <mutex>



namespace ORB_SLAM2
{

class MapPoint;
class MapLine;
class KeyFrame;

class Map
{
public:
    Map();

    void AddKeyFrame(KeyFrame* pKF);
    void EraseKeyFrame(KeyFrame* pKF);
    void InformNewBigChange();
    int GetLastBigChangeIdx();

    // 添加、删除、设置参考地图点，未修改
    void AddMapPoint(MapPoint* pMP);
    void EraseMapPoint(MapPoint* pMP);
    void SetReferenceMapPoints(const std::vector<MapPoint*> &vpMPs);

    // 添加、删除、设置参考地图线，添加
    void AddMapLine(MapLine* pML);
    void EraseMapLine(MapLine* pML);
    void SetReferenceMapLines(const std::vector<MapLine*> &vpMLs);

    // 地图点获取与初始化，未修改
    std::vector<MapPoint*> GetAllMapPoints();
    std::vector<MapPoint*> GetReferenceMapPoints();
    long unsigned int MapPointsInMap();

    // 地图线获取与初始化，添加
    std::vector<MapLine*> GetAllMapLines();
    std::vector<MapLine*> GetReferenceMapLines();
    long unsigned int MapLinesInMap();

    // 关键帧获取与初始化，未修改
    std::vector<KeyFrame*> GetAllKeyFrames();
    long unsigned  KeyFramesInMap();

    long unsigned int GetMaxKFid();

    void clear();

    std::vector<KeyFrame*> mvpKeyFrameOrigins;

    std::mutex mMutexMapUpdate;

    // This avoid that two points are created simultaneously in separate threads (id conflict)
    std::mutex mMutexPointCreation;

    // 避免不同线程中创建线冲突，添加
    std::mutex mMutexLineCreation;

protected:
    std::set<MapPoint*> mspMapPoints;
    // 储存地图线的set容器，添加
    std::set<MapLine*> mspMapLines;

    std::set<KeyFrame*> mspKeyFrames;

    std::vector<MapPoint*> mvpReferenceMapPoints;
    // 储存参考地图线的vector容器，添加
    std::vector<MapLine*> mvpReferenceMapLines;

    long unsigned int mnMaxKFid;

    // Index related to a big change in the map (loop closure, global BA)
    int mnBigChangeIdx;

    std::mutex mMutexMap;
};

} //namespace ORB_SLAM

#endif // MAP_H
