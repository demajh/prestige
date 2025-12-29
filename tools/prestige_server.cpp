#include <prestige/server/server.hpp>
#include <prestige/server/config.hpp>

#include <iostream>
#include <exception>

int main(int argc, char** argv) {
  try {
    // Parse configuration from command line (and optionally config file)
    auto config = prestige::server::Config::LoadFromArgs(argc, argv);

    // Create and run server
    prestige::server::Server server(config);
    server.Run();

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}
