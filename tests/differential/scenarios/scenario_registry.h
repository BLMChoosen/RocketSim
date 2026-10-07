#pragma once
#include <string>
#include <vector>
#include <memory>
#include <map>
#include "scenario.h"

namespace rocketsim_cuda {

class ScenarioRegistry {
public:
    static ScenarioRegistry& Instance();

    void Register(std::shared_ptr<IScenario> scenario);
    void RegisterAlias(const std::string& alias, const std::string& target_name);
    std::shared_ptr<IScenario> Get(const std::string& name) const;
    bool Has(const std::string& name) const;
    std::vector<std::string> GetAllNames() const;
    std::vector<std::shared_ptr<IScenario>> GetAll() const;

private:
    ScenarioRegistry() = default;
    std::map<std::string, std::shared_ptr<IScenario>> scenarios_;
    std::vector<std::string> scenario_order_;
};

// Module-level scenario registration functions
void RegisterBounceScenarios();
void RegisterCarScenarios();
void RegisterAblationScenarios();
void RegisterArenaScenarios();
void RegisterMultiCarScenarios();
void RegisterRandomScenarios();

// Master registration function called at harness startup
void RegisterAllScenarios();

} // namespace rocketsim_cuda
