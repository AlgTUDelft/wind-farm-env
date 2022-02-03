from wind_farm_gym import WindFarmEnv

# Initialize the environment with 3 turbines positioned 750 meters apart in a line
env = WindFarmEnv(turbine_layout=([0, 750, 1500], [0, 0, 0]))

obs = env.reset()
for _ in range(1000):                # Repeat for 1000 steps
    a = env.action_space.sample()    # Choose an action randomly
    obs, reward, _, _ = env.step(a)  # Perform the action
    env.render()                     # Render the environment; remove this line to speed up the process
env.close()
