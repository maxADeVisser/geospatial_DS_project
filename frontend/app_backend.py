intersecting = traj_collection.get_intersecting(area_of_interest)
print(f"Found {len(intersecting)} intersections")

bridge_traj = intersecting.trajectories[0]
bridge_traj.hvplot(
    title=f"Trajectory {bridge_traj.id}",
    frame_width=700,
    frame_height=500,
    line_width=5.0,
    c="NavStatus",
    cmap="Dark2",
)


# Finding ships that depart from Sjöfartsverket (Maritime Administration)
# see: https://movingpandas.github.io/movingpandas-website/2-analysis-examples/ship-data.html#:~:text=%C2%A9%20OpenStreetMap%20contributors-,Finding,-ships%20passing%20under
