#include "depthai/pipeline/ThreadedNode.hpp"


namespace dai
{

void ThreadedNode::start() {
    // Start the thread
    thread = std::thread([this](){
        try {
            run();
        } catch (...){
            // catch anything and stop the node
            running = false;
        }
    });
}

void ThreadedNode::wait() {
    if(thread.joinable()) thread.join();
}

void ThreadedNode::stop() {
    // TBD
    // Sets running to false
    running = false;
    // closes all the queueus, then waits for the thread to join
    for(auto& in : getInputRefs()) {
        in->queue.close();
    }
    // for(auto& rout : getOutputRefs()) {
    // }
    wait();
}

bool ThreadedNode::isRunning() const {
    return running;
}

} // namespace dai
