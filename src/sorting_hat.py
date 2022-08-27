from typing import Any, Dict, List, Tuple, TypeAlias
from ortools.sat.python import cp_model
from ortools.sat.python.cp_model import CpModel, IntVar, BoundedLinearExpression
from pandas import DataFrame

StudentPreferences: TypeAlias = Dict[str, List[str]]
CourseMaxStudents: TypeAlias = Dict[str, int]


def get_example_problem():
    student_preferences: StudentPreferences = {
        "alice": ["alchemy", "sorcery", "arithmancy"],
        "bob": ["alchemy", "sorcery", "magical beasts"],
        "mike truck": ["alchemy", "quidditch", "sorcery"],
        "bobson dugnutt": ["alchemy", "quidditch", "magical beasts"],
    }
    course_max_students: CourseMaxStudents = {
        "alchemy": 2,
        "sorcery": 1,
        "arithmancy": 10,
        "magical beasts": 30,
        "quidditch": 30,
    }
    return student_preferences, course_max_students


class CourseAssignmentVariables:
    def __init__(self, initial_variables: List[Tuple[str, str, cp_model.IntVar]]):
        self.variables: DataFrame = DataFrame(
            data=initial_variables, columns=["student", "course", "variable"]
        )

    def __eq__(self, other: Any):
        if not isinstance(other, self.__class__):
            return False
        # not sure how to meaningfully compare the IntVar objects, so dropping those for now
        return self.variables[["student", "course"]].equals(
            other.variables[["student", "course"]]
        )

    def get_by_student_name(self, name: str) -> List[cp_model.IntVar]:
        student_vars: List[cp_model.IntVar] = self.variables.query(
            f"student == '{name}'"
        )["variable"].to_list()
        return student_vars

    def get_by_course_name(self, name: str) -> List[IntVar]:
        course_vars: List[cp_model.IntVar] = self.variables.query(
            f"course == '{name}'"
        )["variable"].to_list()
        return course_vars

    def get_by_student_name_and_courses(
        self, student_name: str, course_names: List[str]
    ):
        variables: List[IntVar] = self.variables[
            (self.variables["student"] == student_name)
            & (self.variables["course"].isin(course_names))
        ]["variable"].to_list()
        return variables

    def get_all(self) -> List[IntVar]:
        return self.variables['variable'].to_list()


def generate_course_assignment_variables(
    students: StudentPreferences, courses: CourseMaxStudents, model: CpModel
) -> CourseAssignmentVariables:
    student_names: List[str] = list(students.keys())
    course_names: List[str] = list(courses.keys())
    initial_variables: List[Tuple[str, str, cp_model.IntVar]] = [
        (
            student_name,
            course_name,
            model.NewIntVar(0, 1, f"{student_name} in {course_name}"),
        )
        for student_name in student_names
        for course_name in course_names
    ]
    assignments = CourseAssignmentVariables(initial_variables)
    return assignments


def main():
    student_preferences: StudentPreferences
    course_max_students: CourseMaxStudents
    student_preferences, course_max_students = get_example_problem()
    model = cp_model.CpModel()
    assignment_variables: CourseAssignmentVariables = generate_course_assignment_variables(
        student_preferences, course_max_students, model
    )

    exactly_one_course_constraints = generate_constraints_exactly_one_course_per_student(assignment_variables, student_preferences)
    max_students_per_course_constraints = generate_constraints_max_students_per_course(assignment_variables, course_max_students)
    only_preferred_courses_constraints = generate_constraints_only_preferred_courses(assignment_variables, course_max_students, student_preferences)

    all_constraints: List[
        BoundedLinearExpression
    ] = exactly_one_course_constraints + max_students_per_course_constraints + only_preferred_courses_constraints
    for constraint in all_constraints:
        model.Add(constraint)

    total_cost = generate_cost(
        student_preferences, assignment_variables
    )
    model.Minimize(total_cost)

    # see for example: https://developers.google.com/optimization/cp/cp_example
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print(f"Minimum of objective function: {solver.ObjectiveValue()}\n")
        all_assignments: List[IntVar] = assignment_variables.get_all()
        for assignment in all_assignments:
            print(f'{assignment} = {solver.Value(assignment)}')
    else:
        print("No solution found.")


def generate_constraints_only_preferred_courses(assignment_variables: CourseAssignmentVariables, course_max_students: CourseMaxStudents, student_preferences: StudentPreferences) -> List[BoundedLinearExpression]:
    only_preferred_courses_constraints: List[BoundedLinearExpression] = []
    all_course_name_set: set = set(course_max_students.keys())
    student_names: List[str] = list(student_preferences.keys())
    for student_name in student_names:
        student_preferred_course_set: set = set(student_preferences[student_name])
        non_preferred_courses: set = all_course_name_set - student_preferred_course_set
        non_preferred_assign_vars: List[
            IntVar
        ] = assignment_variables.get_by_student_name_and_courses(
            student_name, non_preferred_courses
        )
        for av in non_preferred_assign_vars:
            only_preferred_courses_constraints.append(av == 0)
    return only_preferred_courses_constraints


def generate_constraints_max_students_per_course(assignment_variables: CourseAssignmentVariables, course_max_students: CourseMaxStudents) -> List[BoundedLinearExpression]:
    course_names: list[str] = list(course_max_students.keys())
    max_students_per_course_constraints: List[BoundedLinearExpression] = []
    for course_name in course_names:
        course_max_nr_students: int = course_max_students[course_name]
        variables_for_course: List[IntVar] = assignment_variables.get_by_course_name(
            course_name
        )
        constraint: BoundedLinearExpression = sum(
            variables_for_course
        ) <= course_max_nr_students
        max_students_per_course_constraints.append(constraint)
    return max_students_per_course_constraints


def generate_constraints_exactly_one_course_per_student(assignment_variables: CourseAssignmentVariables, student_preferences: StudentPreferences) -> List[BoundedLinearExpression]:
    student_names: list = list(student_preferences.keys())
    exactly_one_course_constraints: List[BoundedLinearExpression] = []
    for student_name in student_names:
        variables_for_student: List[
            cp_model.IntVar
        ] = assignment_variables.get_by_student_name(student_name)
        constraint: BoundedLinearExpression = sum(variables_for_student) == 1
        exactly_one_course_constraints.append(constraint)
    return exactly_one_course_constraints


def generate_cost(students: StudentPreferences, course_assignments: CourseAssignmentVariables) -> BoundedLinearExpression:
    cost_terms: List[BoundedLinearExpression] = []
    for student_name, course_list in students.items():
        # note that courses in list are ordered by preference
        for preference_index, course in enumerate(course_list):
            assign_var: IntVar = course_assignments.get_by_student_name_and_courses(student_name, [course])[0]
            cost_terms.append(preference_index * assign_var)
    cost: BoundedLinearExpression = sum(cost_terms)
    return cost


