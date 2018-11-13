#pragma once

struct OutputBuffer {
// GLuint 就是正整形，和C里面的unsigned int 一样
  // openGL里面定义的
  GLuint dataBuffer = 0; // 数据缓冲区
  GLuint stateObject = 0;// 目标状态
};
