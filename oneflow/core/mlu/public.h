/*************************************************************************
 * Copyright (C) [2019-2022] by Cambricon, Inc.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/

#ifndef SAMPLES_PUBLIC_PUBLIC_H_
#define SAMPLES_PUBLIC_PUBLIC_H_

#include <iostream>
#include <sstream>
#include <fstream>
#include <stdexcept>
#include <string>
#include "cnrt.h"
#include "cnnl.h"

namespace oneflow {

size_t epoch_elapsed();

#define BIND_MESSAGE(s) \
    std::stringstream _message;                                       \
    _message << __FILE__ << ':' << __LINE__ << " " << std::string(s)  \
             << " [elapsed:" << epoch_elapsed() << " ms]";            \


#define ERROR(s)                                                      \
  {                                                                   \
    BIND_MESSAGE(s)                                                   \
    std::cerr << _message.str() << "\nAborting...\n";                 \
    exit(EXIT_FAILURE);                                               \
  }

#define LOG(s)                                                        \
  {                                                                   \
    BIND_MESSAGE(s)                                                   \
    std::cout << _message.str() << "\n";                              \
  }

std::string cnnlErrorString(cnnlStatus_t status);

void cnnlCheck(cnnlStatus_t result, char const *const func, const char *const file, int const line);

#define CNNL_CHECK(val) cnnlCheck((val), #val, __FILE__, __LINE__)

#endif  // SAMPLES_PUBLIC_PUBLIC_H_

}