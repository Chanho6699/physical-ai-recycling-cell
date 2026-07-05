#include <algorithm>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

#include <control_msgs/action/gripper_command.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <moveit/move_group_interface/move_group_interface.hpp>
#include <moveit/planning_scene_interface/planning_scene_interface.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>

#include <recycling_cell_msgs/action/pick_object.hpp>
#include <recycling_cell_msgs/action/place_object.hpp>

namespace recycling_cell_moveit_manipulation
{

using PickObject = recycling_cell_msgs::action::PickObject;
using GoalHandlePick = rclcpp_action::ServerGoalHandle<PickObject>;

using PlaceObject = recycling_cell_msgs::action::PlaceObject;
using GoalHandlePlace = rclcpp_action::ServerGoalHandle<PlaceObject>;

using GripperCommand = control_msgs::action::GripperCommand;

static constexpr char kPlanningGroup[] = "panda_arm";
// The Panda demo's panda_hand_controller is a
// position_controllers/GripperActionController, which only exposes a
// control_msgs/action/GripperCommand action (not FollowJointTrajectory --
// verified with `ros2 action info /panda_hand_controller/gripper_cmd` and
// `ros2 control list_controllers`). Its single position value drives
// panda_finger_joint1; panda_finger_joint2 mirrors it via a URDF mimic
// joint, so only one value needs to be commanded.
static constexpr char kGripperActionName[] = "/panda_hand_controller/gripper_cmd";

class MoveItManipulationNode : public rclcpp::Node
{
public:
  explicit MoveItManipulationNode(const rclcpp::NodeOptions & options)
  : Node("moveit_manipulation_node", options)
  {
    // automatically_declare_parameters_from_overrides(true) (set in main())
    // already declares a parameter if it was passed as a launch/CLI
    // override, so only declare it here when that hasn't happened.
    declareParameterIfNeeded("enable_planning_scene", true);
    declareParameterIfNeeded("enable_gripper_control", true);
    declareParameterIfNeeded("gripper_open_width", 0.04);
    declareParameterIfNeeded("gripper_close_width", 0.00);

    // The Panda hand's neutral pose quaternion points the gripper "up"
    // (away from the table), so pick/place poses need this override to get
    // a top-down grasp. x=1,y=0,z=0,w=0 is a 180 deg rotation about X from
    // identity -- it is the first candidate to try in RViz; if the fingers
    // still aren't facing the table, try other candidates (e.g. y=1 instead
    // of x=1) via these same launch parameters without touching the code.
    declareParameterIfNeeded("use_fixed_downward_orientation", true);
    declareParameterIfNeeded("downward_qx", 1.0);
    declareParameterIfNeeded("downward_qy", 0.0);
    declareParameterIfNeeded("downward_qz", 0.0);
    declareParameterIfNeeded("downward_qw", 0.0);

    gripper_action_client_ = rclcpp_action::create_client<GripperCommand>(
      this, kGripperActionName);

    // MoveGroupInterface requires a rclcpp::Node::SharedPtr, which is not
    // available yet inside this constructor (shared_from_this() cannot be
    // called until the object is owned by a shared_ptr). It is created in
    // init(), which main() calls right after std::make_shared<>().
    pick_action_server_ = rclcpp_action::create_server<PickObject>(
      this,
      "/manipulation/pick_object",
      std::bind(
        &MoveItManipulationNode::handlePickGoal, this,
        std::placeholders::_1, std::placeholders::_2),
      std::bind(
        &MoveItManipulationNode::handlePickCancel, this,
        std::placeholders::_1),
      std::bind(
        &MoveItManipulationNode::handlePickAccepted, this,
        std::placeholders::_1));

    place_action_server_ = rclcpp_action::create_server<PlaceObject>(
      this,
      "/manipulation/place_object",
      std::bind(
        &MoveItManipulationNode::handlePlaceGoal, this,
        std::placeholders::_1, std::placeholders::_2),
      std::bind(
        &MoveItManipulationNode::handlePlaceCancel, this,
        std::placeholders::_1),
      std::bind(
        &MoveItManipulationNode::handlePlaceAccepted, this,
        std::placeholders::_1));

    RCLCPP_INFO(get_logger(), "moveit_manipulation_node started");
  }

  void init()
  {
    move_group_ = std::make_unique<moveit::planning_interface::MoveGroupInterface>(
      shared_from_this(), kPlanningGroup);

    move_group_->setPlanningTime(5.0);
    move_group_->setMaxVelocityScalingFactor(0.3);
    move_group_->setMaxAccelerationScalingFactor(0.3);

    RCLCPP_INFO(
      get_logger(), "MoveGroupInterface ready for planning group '%s'",
      kPlanningGroup);

    if (get_parameter("enable_planning_scene").as_bool()) {
      setupPlanningScene();
    } else {
      RCLCPP_WARN(
        get_logger(),
        "enable_planning_scene is false; no collision objects were added "
        "to the planning scene");
    }
  }

private:
  template<typename T>
  void declareParameterIfNeeded(const std::string & name, const T & default_value)
  {
    if (!has_parameter(name)) {
      declare_parameter(name, default_value);
    }
  }

  // ---------- planning scene ----------

  moveit_msgs::msg::CollisionObject makeBoxCollisionObject(
    const std::string & id, const std::string & frame_id,
    double x, double y, double z,
    double size_x, double size_y, double size_z)
  {
    moveit_msgs::msg::CollisionObject collision_object;
    collision_object.header.frame_id = frame_id;
    collision_object.id = id;

    shape_msgs::msg::SolidPrimitive primitive;
    primitive.type = primitive.BOX;
    primitive.dimensions = {size_x, size_y, size_z};

    geometry_msgs::msg::Pose box_pose;
    box_pose.orientation.w = 1.0;
    box_pose.position.x = x;
    box_pose.position.y = y;
    box_pose.position.z = z;

    collision_object.primitives.push_back(primitive);
    collision_object.primitive_poses.push_back(box_pose);
    collision_object.operation = collision_object.ADD;

    RCLCPP_INFO(
      get_logger(),
      "Adding collision object '%s': pose=(%.2f, %.2f, %.2f) "
      "size=(%.2f, %.2f, %.2f) frame_id=%s",
      id.c_str(), x, y, z, size_x, size_y, size_z, frame_id.c_str());

    return collision_object;
  }

  void setupPlanningScene()
  {
    const std::string frame_id = move_group_->getPlanningFrame();

    std::vector<moveit_msgs::msg::CollisionObject> collision_objects;
    collision_objects.push_back(
      makeBoxCollisionObject(
        "sorting_table", frame_id, 0.50, 0.00, -0.04, 1.20, 0.80, 0.05));
    collision_objects.push_back(
      makeBoxCollisionObject(
        "plastic_bin", frame_id, 0.35, 0.20, 0.07, 0.12, 0.12, 0.10));
    collision_objects.push_back(
      makeBoxCollisionObject(
        "metal_bin", frame_id, 0.45, 0.20, 0.07, 0.12, 0.12, 0.10));
    collision_objects.push_back(
      makeBoxCollisionObject(
        "paper_bin", frame_id, 0.55, 0.15, 0.07, 0.12, 0.12, 0.10));
    collision_objects.push_back(
      makeBoxCollisionObject(
        "glass_bin", frame_id, 0.60, 0.10, 0.07, 0.12, 0.12, 0.10));
    collision_objects.push_back(
      makeBoxCollisionObject(
        "reject_bin", frame_id, 0.25, 0.20, 0.07, 0.12, 0.12, 0.10));

    planning_scene_interface_.applyCollisionObjects(collision_objects);

    RCLCPP_INFO(
      get_logger(),
      "Planning scene setup complete: %zu collision objects added",
      collision_objects.size());
  }

  // ---------- gripper control ----------

  bool commandGripper(double position)
  {
    if (!gripper_action_client_->wait_for_action_server(std::chrono::seconds(5))) {
      RCLCPP_ERROR(
        get_logger(), "Gripper action server '%s' not available",
        kGripperActionName);
      return false;
    }

    GripperCommand::Goal goal;
    goal.command.position = position;
    goal.command.max_effort = 20.0;

    auto send_goal_future = gripper_action_client_->async_send_goal(goal);
    const auto goal_handle = send_goal_future.get();
    if (!goal_handle) {
      RCLCPP_ERROR(get_logger(), "Gripper goal was rejected");
      return false;
    }

    auto result_future = gripper_action_client_->async_get_result(goal_handle);
    const auto wrapped_result = result_future.get();

    if (wrapped_result.code != rclcpp_action::ResultCode::SUCCEEDED) {
      RCLCPP_ERROR(
        get_logger(), "Gripper action did not succeed (result code %d)",
        static_cast<int>(wrapped_result.code));
      return false;
    }
    return true;
  }

  bool openGripper()
  {
    if (!get_parameter("enable_gripper_control").as_bool()) {
      RCLCPP_INFO(
        get_logger(),
        "[gripper] open (enable_gripper_control=false, "
        "no hardware command sent)");
      return true;
    }

    const double width = get_parameter("gripper_open_width").as_double();
    RCLCPP_INFO(get_logger(), "[gripper] opening to width=%.3f", width);
    return commandGripper(width);
  }

  bool closeGripper()
  {
    if (!get_parameter("enable_gripper_control").as_bool()) {
      RCLCPP_INFO(
        get_logger(),
        "[gripper] close (enable_gripper_control=false, "
        "no hardware command sent)");
      return true;
    }

    const double width = get_parameter("gripper_close_width").as_double();
    RCLCPP_INFO(get_logger(), "[gripper] closing to width=%.3f", width);
    return commandGripper(width);
  }

  // ---------- shared helpers ----------

  static void fixOrientation(geometry_msgs::msg::Pose & pose)
  {
    if (pose.orientation.w == 0.0) {
      pose.orientation.w = 1.0;
    }
  }

  geometry_msgs::msg::Quaternion getDownwardGripperOrientation()
  {
    geometry_msgs::msg::Quaternion q;
    q.x = get_parameter("downward_qx").as_double();
    q.y = get_parameter("downward_qy").as_double();
    q.z = get_parameter("downward_qz").as_double();
    q.w = get_parameter("downward_qw").as_double();
    return q;
  }

  // Overrides pose.orientation with the fixed top-down grasp orientation
  // when use_fixed_downward_orientation is true; otherwise leaves whatever
  // orientation the pose already had (e.g. the goal's own target_pose).
  void applyGripperOrientation(geometry_msgs::msg::Pose & pose)
  {
    if (get_parameter("use_fixed_downward_orientation").as_bool()) {
      pose.orientation = getDownwardGripperOrientation();
    }
  }

  bool moveToPose(const geometry_msgs::msg::Pose & pose)
  {
    std::lock_guard<std::mutex> lock(move_group_mutex_);

    RCLCPP_INFO(
      get_logger(),
      "Planning to pose: x=%.3f, y=%.3f, z=%.3f, q=(%.3f, %.3f, %.3f, %.3f)",
      pose.position.x, pose.position.y, pose.position.z,
      pose.orientation.x, pose.orientation.y, pose.orientation.z,
      pose.orientation.w);

    move_group_->setPoseTarget(pose);

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    const bool planned = static_cast<bool>(move_group_->plan(plan));
    if (!planned) {
      RCLCPP_ERROR(get_logger(), "MoveIt planning failed");
      move_group_->clearPoseTargets();
      return false;
    }

    const bool executed = static_cast<bool>(move_group_->execute(plan));
    if (!executed) {
      RCLCPP_ERROR(get_logger(), "MoveIt execution failed");
    }
    move_group_->clearPoseTargets();
    return executed;
  }

  // ---------- PickObject ----------

  rclcpp_action::GoalResponse handlePickGoal(
    const rclcpp_action::GoalUUID & uuid,
    std::shared_ptr<const PickObject::Goal> goal)
  {
    (void)uuid;
    RCLCPP_INFO(
      get_logger(),
      "PickObject goal received: object_id=%s, class_name=%s",
      goal->object_id.c_str(), goal->class_name.c_str());
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
  }

  rclcpp_action::CancelResponse handlePickCancel(
    const std::shared_ptr<GoalHandlePick> goal_handle)
  {
    (void)goal_handle;
    RCLCPP_WARN(get_logger(), "Received request to cancel pick goal");
    return rclcpp_action::CancelResponse::ACCEPT;
  }

  void handlePickAccepted(const std::shared_ptr<GoalHandlePick> goal_handle)
  {
    std::thread{
      std::bind(&MoveItManipulationNode::executePick, this, goal_handle)
    }.detach();
  }

  bool checkPickCanceled(
    const std::shared_ptr<GoalHandlePick> & goal_handle,
    const std::shared_ptr<PickObject::Result> & result,
    const std::string & object_id)
  {
    if (!goal_handle->is_canceling()) {
      return false;
    }
    result->success = false;
    result->error_code = "CANCELED";
    result->message = "pick of " + object_id + " canceled by client";
    goal_handle->canceled(result);
    return true;
  }

  void abortPick(
    const std::shared_ptr<GoalHandlePick> & goal_handle,
    const std::shared_ptr<PickObject::Result> & result,
    const std::string & object_id)
  {
    result->success = false;
    result->error_code = "MOVEIT_PICK_PLAN_FAILED";
    result->message =
      "MoveIt planning/execution failed for object_id=" + object_id;
    goal_handle->abort(result);
  }

  void executePick(const std::shared_ptr<GoalHandlePick> goal_handle)
  {
    const auto goal = goal_handle->get_goal();
    auto feedback = std::make_shared<PickObject::Feedback>();
    auto result = std::make_shared<PickObject::Result>();

    geometry_msgs::msg::Pose target_pose = goal->target_pose;
    fixOrientation(target_pose);

    geometry_msgs::msg::Pose pre_grasp_pose = target_pose;
    pre_grasp_pose.position.z =
      std::max(target_pose.position.z + 0.35, 0.35);
    applyGripperOrientation(pre_grasp_pose);

    geometry_msgs::msg::Pose approach_pose = target_pose;
    approach_pose.position.z =
      std::max(target_pose.position.z + 0.25, 0.30);
    applyGripperOrientation(approach_pose);

    geometry_msgs::msg::Pose lift_pose = target_pose;
    lift_pose.position.z =
      std::max(target_pose.position.z + 0.45, 0.45);
    applyGripperOrientation(lift_pose);

    if (checkPickCanceled(goal_handle, result, goal->object_id)) {
      return;
    }
    feedback->current_stage = "MOVING_TO_PRE_GRASP";
    feedback->progress = 0.20f;
    goal_handle->publish_feedback(feedback);
    if (!moveToPose(pre_grasp_pose)) {
      abortPick(goal_handle, result, goal->object_id);
      return;
    }

    if (checkPickCanceled(goal_handle, result, goal->object_id)) {
      return;
    }
    feedback->current_stage = "MOVING_TO_GRASP";
    feedback->progress = 0.40f;
    goal_handle->publish_feedback(feedback);
    if (!moveToPose(approach_pose)) {
      abortPick(goal_handle, result, goal->object_id);
      return;
    }

    if (checkPickCanceled(goal_handle, result, goal->object_id)) {
      return;
    }
    feedback->current_stage = "CLOSING_GRIPPER";
    feedback->progress = 0.60f;
    goal_handle->publish_feedback(feedback);
    if (!closeGripper()) {
      result->success = false;
      result->error_code = "GRIPPER_CLOSE_FAILED";
      result->message = "failed to close gripper for object_id=" +
        goal->object_id;
      goal_handle->abort(result);
      return;
    }

    if (checkPickCanceled(goal_handle, result, goal->object_id)) {
      return;
    }
    feedback->current_stage = "LIFTING_OBJECT";
    feedback->progress = 0.80f;
    goal_handle->publish_feedback(feedback);
    if (!moveToPose(lift_pose)) {
      abortPick(goal_handle, result, goal->object_id);
      return;
    }

    feedback->current_stage = "PICK_DONE";
    feedback->progress = 1.00f;
    goal_handle->publish_feedback(feedback);

    result->success = true;
    result->error_code = "";
    result->message = "pick of " + goal->object_id + " completed";
    goal_handle->succeed(result);
  }

  // ---------- PlaceObject ----------

  rclcpp_action::GoalResponse handlePlaceGoal(
    const rclcpp_action::GoalUUID & uuid,
    std::shared_ptr<const PlaceObject::Goal> goal)
  {
    (void)uuid;
    RCLCPP_INFO(
      get_logger(),
      "PlaceObject goal received: object_id=%s, target_bin_id=%s",
      goal->object_id.c_str(), goal->target_bin_id.c_str());
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
  }

  rclcpp_action::CancelResponse handlePlaceCancel(
    const std::shared_ptr<GoalHandlePlace> goal_handle)
  {
    (void)goal_handle;
    RCLCPP_WARN(get_logger(), "Received request to cancel place goal");
    return rclcpp_action::CancelResponse::ACCEPT;
  }

  void handlePlaceAccepted(const std::shared_ptr<GoalHandlePlace> goal_handle)
  {
    std::thread{
      std::bind(&MoveItManipulationNode::executePlace, this, goal_handle)
    }.detach();
  }

  bool checkPlaceCanceled(
    const std::shared_ptr<GoalHandlePlace> & goal_handle,
    const std::shared_ptr<PlaceObject::Result> & result,
    const std::string & object_id)
  {
    if (!goal_handle->is_canceling()) {
      return false;
    }
    result->success = false;
    result->error_code = "CANCELED";
    result->message = "place of " + object_id + " canceled by client";
    goal_handle->canceled(result);
    return true;
  }

  void abortPlace(
    const std::shared_ptr<GoalHandlePlace> & goal_handle,
    const std::shared_ptr<PlaceObject::Result> & result,
    const std::string & object_id)
  {
    result->success = false;
    result->error_code = "MOVEIT_PLACE_PLAN_FAILED";
    result->message =
      "MoveIt planning/execution failed for object_id=" + object_id;
    goal_handle->abort(result);
  }

  void executePlace(const std::shared_ptr<GoalHandlePlace> goal_handle)
  {
    const auto goal = goal_handle->get_goal();
    auto feedback = std::make_shared<PlaceObject::Feedback>();
    auto result = std::make_shared<PlaceObject::Result>();

    geometry_msgs::msg::Pose place_pose = goal->place_pose;
    fixOrientation(place_pose);

    geometry_msgs::msg::Pose pre_place_pose = place_pose;
    pre_place_pose.position.z = std::max(place_pose.position.z + 0.25, 0.45);
    applyGripperOrientation(pre_place_pose);

    geometry_msgs::msg::Pose release_pose = place_pose;
    release_pose.position.z = std::max(place_pose.position.z + 0.15, 0.40);
    applyGripperOrientation(release_pose);

    geometry_msgs::msg::Pose retreat_pose = place_pose;
    retreat_pose.position.z = std::max(place_pose.position.z + 0.35, 0.55);
    applyGripperOrientation(retreat_pose);

    if (checkPlaceCanceled(goal_handle, result, goal->object_id)) {
      return;
    }
    feedback->current_stage = "MOVING_TO_BIN_PRE_PLACE";
    feedback->progress = 0.25f;
    goal_handle->publish_feedback(feedback);
    if (!moveToPose(pre_place_pose)) {
      abortPlace(goal_handle, result, goal->object_id);
      return;
    }

    if (checkPlaceCanceled(goal_handle, result, goal->object_id)) {
      return;
    }
    feedback->current_stage = "MOVING_TO_PLACE_POSE";
    feedback->progress = 0.50f;
    goal_handle->publish_feedback(feedback);
    if (!moveToPose(release_pose)) {
      abortPlace(goal_handle, result, goal->object_id);
      return;
    }

    if (checkPlaceCanceled(goal_handle, result, goal->object_id)) {
      return;
    }
    feedback->current_stage = "OPENING_GRIPPER";
    feedback->progress = 0.75f;
    goal_handle->publish_feedback(feedback);
    if (!openGripper()) {
      result->success = false;
      result->error_code = "GRIPPER_OPEN_FAILED";
      result->message = "failed to open gripper for object_id=" +
        goal->object_id;
      goal_handle->abort(result);
      return;
    }

    if (!moveToPose(retreat_pose)) {
      abortPlace(goal_handle, result, goal->object_id);
      return;
    }

    feedback->current_stage = "PLACE_DONE";
    feedback->progress = 1.00f;
    goal_handle->publish_feedback(feedback);

    result->success = true;
    result->error_code = "";
    result->message = "place of " + goal->object_id + " into " +
      goal->target_bin_id + " completed";
    goal_handle->succeed(result);
  }

  std::unique_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
  std::mutex move_group_mutex_;
  moveit::planning_interface::PlanningSceneInterface planning_scene_interface_;

  rclcpp_action::Server<PickObject>::SharedPtr pick_action_server_;
  rclcpp_action::Server<PlaceObject>::SharedPtr place_action_server_;
  rclcpp_action::Client<GripperCommand>::SharedPtr gripper_action_client_;
};

}  // namespace recycling_cell_moveit_manipulation

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  rclcpp::NodeOptions options;
  options.automatically_declare_parameters_from_overrides(true);

  auto node =
    std::make_shared<recycling_cell_moveit_manipulation::MoveItManipulationNode>(
      options);
  node->init();

  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();

  rclcpp::shutdown();
  return 0;
}
