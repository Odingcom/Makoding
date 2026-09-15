#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

struct Appliance {
    std::string name;
    std::string location;
    double watts;
    double hoursPerDay;
};

double monthlyCost(const Appliance& appliance, double pricePerKwh) {
    return appliance.watts * appliance.hoursPerDay * 30.0 / 1000.0 * pricePerKwh;
}

int main() {
    constexpr double pricePerKwh = 0.15;
    std::vector<Appliance> appliances;
    int count;

    std::cout << "Home Energy Tracker\nHow many appliances? ";
    if (!(std::cin >> count) || count < 0) {
        std::cerr << "Invalid appliance count.\n";
        return 1;
    }

    for (int i = 0; i < count; ++i) {
        Appliance appliance;
        std::cout << "Name, location, watts, hours/day: ";
        std::cin >> appliance.name >> appliance.location
                 >> appliance.watts >> appliance.hoursPerDay;
        if (appliance.watts < 0 || appliance.hoursPerDay < 0) {
            std::cerr << "Power and usage must be non-negative.\n";
            return 1;
        }
        appliances.push_back(appliance);
    }

    std::cout << std::fixed << std::setprecision(2)
              << "\nUsage report (estimated monthly cost at $" << pricePerKwh
              << "/kWh)\n";
    double total = 0.0;
    for (const auto& appliance : appliances) {
        const double kwh = appliance.watts * appliance.hoursPerDay * 30.0 / 1000.0;
        const double cost = kwh * pricePerKwh;
        total += cost;
        std::cout << appliance.name << " @ " << appliance.location
                  << ": " << kwh << " kWh, $" << cost << '\n';
    }
    std::cout << "Total estimated monthly cost: $" << total << '\n';
    std::cout << "Note: an Android app should obtain location and live power data"
                 " through platform permissions and compatible sensors.\n";
}