#include <algorithm>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

#include <geometry_msgs/msg/pose.hpp>
#include <moveit/move_group_interface/move_group_interface.hpp>

#include <recycling_cell_msgs/action/pick_object.hpp>
#include <recycling_cell_msgs/action/place_object.hpp>

namespace recycling_cell_moveit_manipulation
{

using PickObject = recycling_cell_msgs::action::PickObject;
using GoalHandlePick = rclcpp_action::ServerGoalHandle<PickObject>;

using PlaceObject = recycling_cell_msgs::action::PlaceObject;
using GoalHandlePlace = rclcpp_action::ServerGoalHandle<PlaceObject>;

static constexpr char kPlanningGroup[] = "panda_arm";

class MoveItManipulationNode : public rclcpp::Node
{
public:
  explicit MoveItManipulationNode(const rclcpp::NodeOptions & options)
  : Node("moveit_manipulation_node", options)
  {
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
  }

private:
  // ---------- shared helpers ----------

  static void fixOrientation(geometry_msgs::msg::Pose & pose)
  {
    if (pose.orientation.w == 0.0) {
      pose.orientation.w = 1.0;
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

    geometry_msgs::msg::Pose approach_pose = target_pose;
    approach_pose.position.z =
      std::max(target_pose.position.z + 0.25, 0.30);

    geometry_msgs::msg::Pose lift_pose = target_pose;
    lift_pose.position.z =
      std::max(target_pose.position.z + 0.45, 0.45);

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
    RCLCPP_INFO(
      get_logger(),
      "[pick] closing gripper (no gripper hardware control yet)");

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

    geometry_msgs::msg::Pose release_pose = place_pose;
    release_pose.position.z = std::max(place_pose.position.z + 0.15, 0.40);

    geometry_msgs::msg::Pose retreat_pose = place_pose;
    retreat_pose.position.z = std::max(place_pose.position.z + 0.35, 0.55);

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
    RCLCPP_INFO(
      get_logger(),
      "[place] opening gripper (no gripper hardware control yet)");

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

  rclcpp_action::Server<PickObject>::SharedPtr pick_action_server_;
  rclcpp_action::Server<PlaceObject>::SharedPtr place_action_server_;
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
