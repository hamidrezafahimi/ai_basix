import envGUICreation as eguic

if __name__ == '__main__':

    env = eguic.GymGraphicalFrozenLake((8,8))
    env.make()

    state = env.reset()
    done = False
    score = 0

    while not done:

        action = env.env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        
        score+=reward

        if done:
            break

    print('Score:{}'.format(score))

    env.close()