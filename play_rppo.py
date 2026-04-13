import argparse
import cv2
import numpy as np

from obelix import OBELIX
import rppo


ACTIONS=("L45","L22","FW","R22","R45")


def reset_agent_state():
    if hasattr(rppo,"_STEP"):
        rppo._STEP=0
    if hasattr(rppo,"_reset_hidden"):
        rppo._reset_hidden()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--scaling_factor",type=int,default=5)
    p.add_argument("--arena_size",type=int,default=500)
    p.add_argument("--max_steps",type=int,default=1000)
    p.add_argument("--wall_obstacles",action="store_true")
    p.add_argument("--difficulty",type=int,default=3)
    p.add_argument("--box_speed",type=int,default=2)
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--episodes",type=int,default=1)
    p.add_argument("--delay",type=int,default=30)
    args=p.parse_args()

    total_scores=[]

    for ep in range(args.episodes):
        seed=args.seed+ep
        env=OBELIX(
            scaling_factor=args.scaling_factor,
            arena_size=args.arena_size,
            max_steps=args.max_steps,
            wall_obstacles=args.wall_obstacles,
            difficulty=args.difficulty,
            box_speed=args.box_speed,
            seed=seed,
        )
        obs=env.reset(seed=seed)
        reset_agent_state()

        done=False
        total=0.0
        step=0

        while not done:
            env.render_frame()
            action=rppo.policy(obs,np.random.default_rng(seed))
            obs,reward,done=env.step(action,render=True)
            total+=float(reward)
            step+=1

            print(f"ep={ep+1}/{args.episodes} step={step} action={action} reward={reward:.1f} total={total:.1f}")

            key=cv2.waitKey(args.delay)&255
            if key in (ord('q'),27):
                done=True
                break

        total_scores.append(total)
        print(f"Episode {ep+1} done, score={total:.1f}")

    if total_scores:
        print(f"Mean score={float(np.mean(total_scores)):.1f}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__=="__main__":
    main()
