#include "PTNET.h"

int main(int argc, char const *argv[])
{
    // Create a new Net.
    auto net = std::make_shared<Net>();
    net.
    torch::load(net, "net.pt");

    // Get Instance image.
    torch::data::datasets::MNIST database("./data");
    auto size = database.size().value();
    std::cout << "Size of database is: " << size << std::endl;
    
    {
        auto example = database.get(0);
        auto result = net->forward(example.data);
        std::cout << "Result is: " << std::endl << result << std::endl;
    }
    
    // for(uint i = 0 ; i < size; ++i){
    //     auto example = database.get(i);
    //     bool comparison = example.target.equal(net->forward(example.data));
    //     if(!comparison)
    //         std::cout << "Wrong recognition at index: " << i << std::endl;
    // }
    std::cout << "Job finished." << std::endl;
    return 0;
}
